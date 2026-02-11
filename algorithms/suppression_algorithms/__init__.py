from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.suppression_algorithms.greedy import GreedyAlgorithm
from algorithms.suppression_algorithms.lp_suppression import LPSuppressionAlgorithm

SUPPRESSION_ALGORITHM_REGISTRY = {
    GreedyAlgorithm.name: GreedyAlgorithm,
    LPSuppressionAlgorithm.name: LPSuppressionAlgorithm,
}

__all__ = [
    "SuppressionAlgorithm",
    "GreedyAlgorithm",
    "LPSuppressionAlgorithm",
    "SUPPRESSION_ALGORITHM_REGISTRY",
]
