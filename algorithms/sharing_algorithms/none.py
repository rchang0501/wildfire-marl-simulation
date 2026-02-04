from algorithms.sharing_algorithms.algorithm_base import SharingAlgorithm


class NoSharingAlgorithm(SharingAlgorithm):
    """Sharing algorithm that never transfers units."""

    name = "none"

    def decide_transfers(self, multi_env, rng) -> list[tuple[int, int]]:
        return []
