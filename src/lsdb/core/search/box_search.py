from __future__ import annotations

import nested_pandas as npd
import numpy as np
from hats.catalog import TableProperties
from hats.pixel_math.box_filter import wrap_ra_angles
from hats.pixel_math.validators import validate_box
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import HCCatalogTypeVar


class BoxSearch(AbstractSearch):
    """Perform a box search to filter the catalog. This type of search is used for a
    range of right ascension or declination, where the right ascension edges follow
    great arc circles and the declination edges follow small arc circles.

    Filters to points within the ra / dec region, specified in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, ra: tuple[float, float], dec: tuple[float, float], fine: bool = True):
        super().__init__(fine)
        ra = tuple(wrap_ra_angles(ra)) if ra else None
        validate_box(ra, dec)
        self.ra, self.dec = ra, dec

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> MOC:
        """Filters catalog pixels according to the box"""
        return hc_structure.filter_by_box(self.ra, self.dec)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return box_filter(frame, self.ra, self.dec, metadata)


def box_filter(
    data_frame: npd.NestedFrame,
    ra: tuple[float, float],
    dec: tuple[float, float],
    metadata: TableProperties,
) -> npd.NestedFrame:
    """Filters a dataframe to only include points within the specified box region.

    Args:
        data_frame (npd.NestedFrame): DataFrame containing points in the sky
        ra (tuple[float, float]): Right ascension range, in degrees
        dec (tuple[float, float]): Declination range, in degrees
        metadata (hc.catalog.Catalog): hats `Catalog` with catalog_info that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the box region.
    """
    ra_values = data_frame[metadata.ra_column].to_numpy()
    dec_values = data_frame[metadata.dec_column].to_numpy()
    wrapped_ra = wrap_ra_angles(ra_values)
    mask_ra = _create_ra_mask(ra, wrapped_ra)
    mask_dec = (dec[0] <= dec_values) & (dec_values <= dec[1])
    data_frame = data_frame.iloc[mask_ra & mask_dec]
    return data_frame


def _create_ra_mask(ra: tuple[float, float], values: np.ndarray) -> np.ndarray:
    """Creates the mask to filter right ascension values. If this range crosses
    the discontinuity line (0 degrees), we have a branched logical operation."""
    if ra[0] == ra[1]:
        return np.ones(len(values), dtype=bool)
    if ra[0] < ra[1]:
        mask = (values >= ra[0]) & (values <= ra[1])
    else:
        mask = ((values >= ra[0]) & (values <= 360)) | ((values >= 0) & (values <= ra[1]))
    return mask
