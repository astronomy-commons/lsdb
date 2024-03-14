from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import hipscat as hc
import pandas as pd
from hipscat.pixel_math import HealpixPixel


# pylint: disable=too-many-instance-attributes, too-many-arguments
class AbstractSearch(ABC):
    """Abstract class used to write a reusable search query.

    These consist of two parts:

        - partition search - a (usually) coarse method of restricting
          the search space to just the partitions(/pixels) of interest
        - point search - a (usally) finer grained method to find
          individual rows matching the query terms.
    """

    @abstractmethod
    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""

    @abstractmethod
    def search_points(self, frame: pd.DataFrame, metadata: hc.catalog.Catalog) -> pd.DataFrame:
        """Determine the search results within a data frame"""
