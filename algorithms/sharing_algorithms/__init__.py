from algorithms.sharing_algorithms.algorithm_base import SharingAlgorithm
from algorithms.sharing_algorithms.none import NoSharingAlgorithm
from algorithms.sharing_algorithms.periodic_transfer import PeriodicTransferSharingAlgorithm

SHARING_ALGORITHM_REGISTRY = {
    NoSharingAlgorithm.name: NoSharingAlgorithm,
    PeriodicTransferSharingAlgorithm.name: PeriodicTransferSharingAlgorithm,
}

__all__ = [
    "SharingAlgorithm",
    "NoSharingAlgorithm",
    "PeriodicTransferSharingAlgorithm",
    "SHARING_ALGORITHM_REGISTRY",
]
