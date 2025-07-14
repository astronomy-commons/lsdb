from enum import Enum

from lsdb.core.crossmatch.bounded_kdtree_match import BoundedKdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


class BuiltInCrossmatchAlgorithm(str, Enum):
    """Cross-matching algorithms included in lsdb"""

    KD_TREE = "kd_tree"
    BOUNDED_KD_TREE = "bounded_kd_tree"


builtin_crossmatch_algorithms = {
    BuiltInCrossmatchAlgorithm.KD_TREE: KdTreeCrossmatch,
    BuiltInCrossmatchAlgorithm.BOUNDED_KD_TREE: BoundedKdTreeCrossmatch,
}


def is_builtin_algorithm(algorithm_type) -> bool:
    """Check if a given algorithm is built-in."""
    return (
        algorithm_type in builtin_crossmatch_algorithms
        or algorithm_type in builtin_crossmatch_algorithms.values()
    )
