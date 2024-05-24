from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.loaders.hipscat.abstract_catalog_loader import HCCatalogTypeVar


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels: List[Tuple[int, int]]):
        self.pixels = [HealpixPixel(o, p) for o, p in set(pixels)]

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_from_pixel_list(self.pixels)

    def search_points(self, frame: pd.DataFrame, metadata: CatalogInfo) -> pd.DataFrame:
        return frame
