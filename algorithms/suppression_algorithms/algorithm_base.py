import json
import os
from abc import ABC, abstractmethod
import numpy as np

from environment.jurisdiction_env import JurisdictionEnv


class SuppressionAlgorithm(ABC):
    """Superclass for suppression algorithms.

    Operates on a single JurisdictionEnv. Returns (num_units, 2) array of (dx, dy).
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
    def actions(self, jenv: JurisdictionEnv, rng: np.random.Generator) -> np.ndarray:
        """Return (jenv.num_units, 2) array of (dx, dy)."""
        raise NotImplementedError()

    def get_actions(self, jenv: JurisdictionEnv, rng: np.random.Generator) -> np.ndarray:
        actions = np.asarray(self.actions(jenv, rng), dtype=int)
        expected_shape = (jenv.num_units, 2)
        if actions.shape != expected_shape:
            raise ValueError(f"actions shape {actions.shape} != {expected_shape}")
        return actions
