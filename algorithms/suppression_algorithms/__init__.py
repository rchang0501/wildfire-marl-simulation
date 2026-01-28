from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.suppression_algorithms.greedy import GreedyAlgorithm

SUPPRESSION_ALGORITHM_REGISTRY = {
    GreedyAlgorithm.name: GreedyAlgorithm,
}

__all__ = [
    "SuppressionAlgorithm",
    "GreedyAlgorithm",
    "SUPPRESSION_ALGORITHM_REGISTRY",
]
