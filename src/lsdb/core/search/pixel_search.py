from __future__ import annotations

from typing import List

import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.filter import get_filtered_pixel_list
from hipscat.pixel_tree.pixel_tree import PixelTree

from lsdb.core.search.abstract_search import AbstractSearch


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels: List[HealpixPixel]):
        self.pixels = list(set(pixels))

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        pixel_tree = PixelTree.from_healpix(pixels)
        filter_pixel_tree = PixelTree.from_healpix(self.pixels)
        return get_filtered_pixel_list(pixel_tree, filter_pixel_tree)

    def search_points(self, frame: pd.DataFrame, metadata: CatalogInfo) -> pd.DataFrame:
        return frame
