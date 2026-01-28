import numpy as np
from algorithms.sharing_algorithms.algorithm_base import SharingAlgorithm


class NoSharingAlgorithm(SharingAlgorithm):
    """
    Sharing algorithm that assigns no units.
    """

    name = "none"

    def assignments(self, env, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        assigned_mask = np.zeros((env.num_units_total,), dtype=bool)
        actions = self._default_actions(env)
        return assigned_mask, actions
