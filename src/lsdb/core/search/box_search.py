from __future__ import annotations

from typing import List, Tuple

import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.box_filter import filter_pixels_by_box, transform_radec, wrap_angles
from hipscat.pixel_math.validators import validate_box_search
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.core.search.abstract_search import AbstractSearch


class BoxSearch(AbstractSearch):
    """Perform a box search to filter the catalog. This type of search is used for a
    range of ra or dec. If both, a polygonal search is the better option.

    Filters to points within the ra / dec region, specified in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(
        self,
        metadata: hc.catalog.Catalog,
        ra: Tuple[float, float] | None = None,
        dec: Tuple[float, float] | None = None,
    ):
        validate_box_search(ra, dec)

        self.ra, self.dec = transform_radec(ra, dec)
        self.metadata = metadata

    def _create_ra_mask(self, values) -> np.ndarray:
        """Creates the mask to filter right ascension values. If this range crosses
        the discontinuity line (0 degrees), we have a branched logical operation.

        Returns:
            A numpy array of bool, where each True value indicates that the
            point belongs to the right ascension range, False indicates otherwise.
        """
        if self.ra is None:
            raise ValueError("No right ascension range defined")
        if self.ra[0] <= self.ra[1]:
            mask = np.logical_and(self.ra[0] <= values, values <= self.ra[1])
        else:
            mask = np.logical_or(
                np.logical_and(self.ra[0] <= values, values <= 360),
                np.logical_and(0 <= values, values <= self.ra[1]),
            )
        return mask

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""
        pixel_tree = PixelTreeBuilder.from_healpix(pixels)
        return filter_pixels_by_box(pixel_tree, self.ra, self.dec)

    def search_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        mask = None
        if self.ra is not None:
            ra_column = self.metadata.catalog_info.ra_column
            wrapped_ra = wrap_angles(frame[ra_column].compute())
            ra_values = np.array(wrapped_ra)
            mask = self._create_ra_mask(ra_values)
        elif self.dec is not None:
            dec_column = self.metadata.catalog_info.dec_column
            dec_values = np.array(frame[dec_column].compute())
            mask = np.logical_and(self.dec[0] <= dec_values, dec_values <= self.dec[1])
        if mask is not None:
            frame = frame.iloc[mask]
        return frame
