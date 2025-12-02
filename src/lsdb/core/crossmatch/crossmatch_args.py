from dataclasses import dataclass

import nested_pandas as npd
from hats.catalog.dataset.table_properties import TableProperties


@dataclass
class CrossmatchArgs:
    """Holds the partition and pixel information to be
    used in the crossmatch algorithm."""

    left_df: npd.NestedFrame
    """Partition from the left catalog"""
    right_df: npd.NestedFrame
    """Partition from the right catalog"""
    left_order: int
    """Left pixel order"""
    left_pixel: int
    """Left pixel number"""
    right_order: int
    """Right pixel order"""
    right_pixel: int
    """Right pixel number"""
    left_catalog_info: TableProperties
    """Catalog info for the left partition"""
    right_catalog_info: TableProperties
    """Catalog info for the right partition"""
    right_margin_catalog_info: TableProperties | None
    """Catalog info for the right margin partition"""
