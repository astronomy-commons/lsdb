from dataclasses import dataclass

import nested_pandas as npd
import numpy as np
import numpy.typing as npt
from hats.catalog.dataset.table_properties import TableProperties


@dataclass
class CrossmatchArgs:
    """Holds the partition and pixel information to be
    used in the crossmatch algorithm."""

    left_df: npd.NestedFrame
    """Partition from the left catalog"""
    right_df: npd.NestedFrame
    """Partition from the right catalog"""
    left_order: int | None
    """Left pixel order"""
    left_pixel: int | None
    """Left pixel number"""
    right_order: int | None
    """Right pixel order"""
    right_pixel: int | None
    """Right pixel number"""
    left_catalog_info: TableProperties
    """Catalog info for the left partition"""
    right_catalog_info: TableProperties
    """Catalog info for the right partition"""
    right_margin_catalog_info: TableProperties | None
    """Catalog info for the right margin partition"""
    right_native_mask: npt.NDArray[np.bool_] | None = None
    """Rows in ``right_df`` belonging to the primary right partition for this aligned pixel."""
    left_native_len: int | None = None
    """Number of leading rows of ``left_df`` belonging to the aligned pixel. Only set for outer joins.

    Any trailing rows are *neighbor* rows: the left margin partition's rows and, when the left
    partition is coarser than the aligned pixel, that partition's own rows just outside the aligned
    pixel. In sky with no left partition this is 0 and every row is a neighbor. Neighbor rows are
    used only to keep the right rows they match from being emitted as unmatched; they are emitted by
    their own pixel's task, never by this one.
    """
