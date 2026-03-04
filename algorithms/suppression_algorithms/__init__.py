from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.suppression_algorithms.greedy import GreedyAlgorithm
from algorithms.suppression_algorithms.lp_suppression import LPSuppressionAlgorithm
from algorithms.suppression_algorithms.rl_suppression import RLSuppressionAlgorithm

SUPPRESSION_ALGORITHM_REGISTRY = {
    GreedyAlgorithm.name: GreedyAlgorithm,
    LPSuppressionAlgorithm.name: LPSuppressionAlgorithm,
    RLSuppressionAlgorithm.name: RLSuppressionAlgorithm,
}

__all__ = [
    "SuppressionAlgorithm",
    "GreedyAlgorithm",
    "LPSuppressionAlgorithm",
    "RLSuppressionAlgorithm",
    "SUPPRESSION_ALGORITHM_REGISTRY",
]
