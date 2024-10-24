from __future__ import annotations

from typing import Tuple

import nested_pandas as npd
import numpy as np
from hats.catalog import TableProperties
from hats.pixel_math.box_filter import generate_box_moc, wrap_ra_angles
from hats.pixel_math.validators import validate_box_search
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch


class BoxSearch(AbstractSearch):
    """Perform a box search to filter the catalog. This type of search is used for a
    range of ra or dec (one or the other). If both, a polygonal search should be used.

    Filters to points within the ra / dec region, specified in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(
        self,
        ra: Tuple[float, float] | None = None,
        dec: Tuple[float, float] | None = None,
        fine: bool = True,
    ):
        super().__init__(fine)
        ra = tuple(wrap_ra_angles(ra)) if ra else None
        validate_box_search(ra, dec)
        self.ra, self.dec = ra, dec

    def generate_search_moc(self, max_order: int) -> MOC:
        return generate_box_moc(self.ra, self.dec, max_order)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return box_filter(frame, self.ra, self.dec, metadata)


def box_filter(
    data_frame: npd.NestedFrame,
    ra: Tuple[float, float] | None,
    dec: Tuple[float, float] | None,
    metadata: TableProperties,
) -> npd.NestedFrame:
    """Filters a dataframe to only include points within the specified box region.

    Args:
        data_frame (npd.NestedFrame): DataFrame containing points in the sky
        ra (Tuple[float, float]): Right ascension range, in degrees
        dec (Tuple[float, float]): Declination range, in degrees
        metadata (hc.catalog.Catalog): hats `Catalog` with catalog_info that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the box region.
    """
    mask = np.ones(len(data_frame), dtype=bool)
    if ra is not None:
        ra_values = data_frame[metadata.ra_column]
        wrapped_ra = np.asarray(wrap_ra_angles(ra_values))
        mask_ra = _create_ra_mask(ra, wrapped_ra)
        mask = np.logical_and(mask, mask_ra)
    if dec is not None:
        dec_values = data_frame[metadata.dec_column].to_numpy()
        mask_dec = np.logical_and(dec[0] <= dec_values, dec_values <= dec[1])
        mask = np.logical_and(mask, mask_dec)
    data_frame = data_frame.iloc[mask]
    return data_frame


def _create_ra_mask(ra: Tuple[float, float], values: np.ndarray) -> np.ndarray:
    """Creates the mask to filter right ascension values. If this range crosses
    the discontinuity line (0 degrees), we have a branched logical operation."""
    if ra[0] <= ra[1]:
        mask = np.logical_and(ra[0] <= values, values <= ra[1])
    else:
        mask = np.logical_or(
            np.logical_and(ra[0] <= values, values <= 360),
            np.logical_and(0 <= values, values <= ra[1]),
        )
    return mask
