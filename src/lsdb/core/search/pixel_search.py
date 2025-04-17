from __future__ import annotations

from typing import TYPE_CHECKING

import nested_pandas as npd
import pandas as pd
from hats.pixel_math import HealpixPixel, get_healpix_pixel, spatial_index

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels: tuple[int, int] | HealpixPixel | list[tuple[int, int] | HealpixPixel]):
        super().__init__(fine=False)
        if isinstance(pixels, tuple):
            self.pixels = [get_healpix_pixel(pixels)]
        elif isinstance(pixels, HealpixPixel):
            self.pixels = [pixels]
        elif pd.api.types.is_list_like(pixels):
            if len(pixels) == 0:
                raise ValueError("Some pixels required for PixelSearch")
            self.pixels = [get_healpix_pixel(pix) for pix in pixels]
        else:
            raise ValueError("Unsupported input for PixelSearch")

    @classmethod
    def from_radec(cls, ra: float | list[float], dec: float | list[float]) -> PixelSearch:
        """Create a pixel search region, based on radec points.

        Args:
            ra (float|list[float]): celestial coordinates, right ascension in degrees
            dec (float|list[float]): celestial coordinates, declination in degrees
        """
        pixels = list(spatial_index.compute_spatial_index(ra, dec))
        pixels = [(spatial_index.SPATIAL_INDEX_ORDER, pix) for pix in pixels]
        return cls(pixels)

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_from_pixel_list(self.pixels)

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        return frame
