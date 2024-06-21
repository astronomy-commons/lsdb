from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from mocpy import MOC

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


# pylint: disable=too-many-instance-attributes, too-many-arguments
class AbstractSearch(ABC):
    """Abstract class used to write a reusable search query.

    These consist of two parts:
        - partition search - a (usually) coarse method of restricting
          the search space to just the partitions(/pixels) of interest
        - point search - a (usally) finer grained method to find
          individual rows matching the query terms.
    """

    def __init__(self, fine: bool = True):
        self.fine = fine

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        """Filters the hispcat catalog object to the partitions included in the search"""
        if len(hc_structure.get_healpix_pixels()) == 0:
            return hc_structure
        max_order = hc_structure.get_max_coverage_order()
        search_moc = self.generate_search_moc(max_order)
        return hc_structure.filter_by_moc(search_moc)

    def generate_search_moc(self, max_order: int) -> MOC:
        """Determine the target partitions for further filtering."""
        raise NotImplementedError(
            "Search Class must implement `generate_search_moc` method or overwrite `filter_hc_catalog`"
        )

    @abstractmethod
    def search_points(self, frame: pd.DataFrame, metadata: CatalogInfo) -> pd.DataFrame:
        """Determine the search results within a data frame"""
