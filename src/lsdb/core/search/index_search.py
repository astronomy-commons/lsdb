from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from hipscat.catalog.index.index_catalog import IndexCatalog

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.loaders.hipscat.abstract_catalog_loader import HCCatalogTypeVar


class IndexSearch(AbstractSearch):
    """Find rows by ids (or other value indexed by a catalog index).

    Filters partitions in the catalog to those that could contain the ids requested.
    Filters to points that have matching values in the id field.

    NB: This requires a previously-computed catalog index table.
    """

    def __init__(self, ids, catalog_index: IndexCatalog):
        self.ids = ids
        self.catalog_index = catalog_index

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        healpix_pixels = self.catalog_index.loc_partitions(self.ids)
        return hc_structure.filter_from_pixel_list(healpix_pixels)

    def search_points(self, frame: pd.DataFrame, _) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        return frame[frame[self.catalog_index.catalog_info.indexing_column].isin(self.ids)]
