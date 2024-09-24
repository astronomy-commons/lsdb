from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import nested_pandas as npd
from hats.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels: List[Tuple[int, int]]):
        super().__init__(fine=False)
        self.pixels = [HealpixPixel(o, p) for o, p in set(pixels)]

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_from_pixel_list(self.pixels)

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        return frame
