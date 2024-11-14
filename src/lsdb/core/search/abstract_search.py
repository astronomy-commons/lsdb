from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import nested_pandas as npd
from hats.catalog import TableProperties
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

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> MOC:
        """Determine the target partitions for further filtering."""
        raise NotImplementedError("Search Class must implement `filter_hc_catalog` method")

    @abstractmethod
    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
