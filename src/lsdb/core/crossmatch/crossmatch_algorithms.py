from enum import Enum


class BuiltInCrossmatchAlgorithm(str, Enum):
    """Cross-matching algorithms included in lsdb"""

    KD_TREE = "kd_tree"
