import json
import os
from abc import ABC, abstractmethod


class SharingAlgorithm(ABC):
    """Superclass for sharing algorithms.

    Operates on a MultiJurisdictionEnv. Decides which units to transfer
    between jurisdictions and optionally provides steering overrides to
    move units toward their jurisdiction center before transfer.
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
    def decide_transfers(self, multi_env, rng) -> list[tuple[int, int]]:
        """Return list of (unit_id, target_juris) pairs to transfer."""
        raise NotImplementedError()

    def get_steering_actions(self, multi_env, rng) -> dict[int, tuple[int, int]]:
        """Return {unit_id: (dx, dy)} overrides for units being steered to center.

        Default: empty dict (no steering).
        """
        return {}
