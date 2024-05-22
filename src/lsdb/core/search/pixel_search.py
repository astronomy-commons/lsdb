from __future__ import annotations

from typing import List

import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels):
        self.pixels = set(pixels)

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        return [p for p in pixels if p in self.pixels]

    def search_points(self, frame: pd.DataFrame, metadata: CatalogInfo) -> pd.DataFrame:
        return frame
