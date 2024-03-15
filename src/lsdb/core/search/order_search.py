from typing import List

import pandas as pd
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch


class OrderSearch(AbstractSearch):
    """Find rows by HEALPix order.

    Filters partitions in the catalog to those that are in the orders specified.
    Returns all points in those partitions
    """

    def __init__(self, min_order: int, max_order: int):
        if min_order > max_order:
            raise ValueError("The minimum order should be less than or equal to maximum order.")

        self.min_order = min_order
        self.max_order = max_order

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""
        return [pixel for pixel in pixels if self.min_order <= pixel.order <= self.max_order]

    def search_points(self, frame: pd.DataFrame, _) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        return frame
