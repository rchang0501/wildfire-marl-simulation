import json
import os
from abc import ABC, abstractmethod
import numpy as np


class SuppressionAlgorithm(ABC):
    """
    Superclass for suppression algorithms.
    Algorithms decide actions for units not already assigned by sharing.
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
    def actions(
        self,
        env,
        rng: np.random.Generator,
        assigned_mask: np.ndarray,
        assigned_actions: np.ndarray,
    ) -> np.ndarray:
        """
        Return actions array of shape (N,3) with entries (tag,a,b).
        The suppression algorithm should only decide actions for units
        where assigned_mask is False.
        """
        raise NotImplementedError()

    def get_actions(
        self,
        env,
        rng: np.random.Generator,
        assigned_mask: np.ndarray,
        assigned_actions: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(self.actions(env, rng, assigned_mask, assigned_actions), dtype=int)
        expected_shape = (env.num_units_total, 3)
        if actions.shape != expected_shape:
            raise ValueError(f"actions shape {actions.shape} != {expected_shape}")

        actions[assigned_mask] = assigned_actions[assigned_mask]

        tags = actions[:, 0]
        in_transit = env.unit_positions[:, 1] < 0
        valid_tags = np.isin(tags, [0, 1, 2])
        if not np.all(valid_tags[~in_transit]):
            raise ValueError("Invalid action tag for a unit not in transit.")
        return actions

    def _default_actions(self, env) -> np.ndarray:
        actions = np.zeros((env.num_units_total, 3), dtype=int)
        actions[:, 0] = 2  # stay by default
        return actions
