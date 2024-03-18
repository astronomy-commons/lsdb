from __future__ import annotations

from typing import List

import pandas as pd
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch


class OrderSearch(AbstractSearch):
    """Filter the catalog by HEALPix order.

    Filters partitions in the catalog to those that are in the orders specified.
    Does not filter points inside those partitions.
    """

    def __init__(self, min_order: int = 0, max_order: int | None = None):
        if max_order and min_order > max_order:
            raise ValueError("The minimum order should be lower than or equal to the maximum order")
        self.min_order = min_order
        self.max_order = max_order

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""
        max_catalog_order = max(pixel.order for pixel in pixels)
        max_order = max_catalog_order if self.max_order is None else self.max_order
        if self.min_order > max_order:
            raise ValueError("The minimum order is higher than the catalog's maximum order")
        return [pixel for pixel in pixels if self.min_order <= pixel.order <= max_order]

    def search_points(self, frame: pd.DataFrame, _) -> pd.DataFrame:
        """Determine the search results within a data frame."""
        return frame
