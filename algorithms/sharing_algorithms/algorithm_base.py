import json
import os
from abc import ABC, abstractmethod
import numpy as np


class SharingAlgorithm(ABC):
    """
    Superclass for sharing algorithms.
    Algorithms choose which units they control and provide actions for them.
    """

    def __init__(self, param_dir: str | None = None, params: dict | None = None):
        self.param_dir = param_dir or ""
        if self.param_dir and not os.path.isdir(self.param_dir):
            raise ValueError(f"param_dir does not exist: {self.param_dir}")
        self.params = self._load_params()
        if params:
            self.params.update(params)

    def _load_params(self) -> dict:
        if not self.param_dir:
            return {}
        params_path = os.path.join(self.param_dir, "params.json")
        if os.path.isfile(params_path):
            with open(params_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    @abstractmethod
    def assignments(self, env, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (assigned_mask, actions) where:
          assigned_mask: (N,) bool array, True if this algorithm controls the unit.
          actions: (N,3) int array of (tag,a,b), used for assigned units.
        """
        raise NotImplementedError()

    def get_assignments(self, env, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        assigned_mask, actions = self.assignments(env, rng)
        assigned_mask = np.asarray(assigned_mask, dtype=bool)
        actions = np.asarray(actions, dtype=int)

        expected_mask_shape = (env.num_units_total,)
        expected_actions_shape = (env.num_units_total, 3)
        if assigned_mask.shape != expected_mask_shape:
            raise ValueError(f"assigned_mask shape {assigned_mask.shape} != {expected_mask_shape}")
        if actions.shape != expected_actions_shape:
            raise ValueError(f"actions shape {actions.shape} != {expected_actions_shape}")

        tags = actions[:, 0]
        in_transit = env.unit_positions[:, 1] < 0
        valid_tags = np.isin(tags, [0, 1, 2])
        if not np.all(valid_tags[assigned_mask & ~in_transit]):
            raise ValueError("Invalid action tag for an assigned unit not in transit.")
        return assigned_mask, actions

    def _default_actions(self, env) -> np.ndarray:
        actions = np.zeros((env.num_units_total, 3), dtype=int)
        actions[:, 0] = 2  # stay by default
        return actions
