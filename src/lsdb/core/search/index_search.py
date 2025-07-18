from __future__ import annotations

from typing import Any

import nested_pandas as npd
import numpy as np
from hats.catalog.index.index_catalog import IndexCatalog as HCIndexCatalog

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import HCCatalogTypeVar


class IndexSearch(AbstractSearch):
    """Find rows by column values using HATS index catalogs."""

    values: dict[str, Any]
    """Mapping of field name to the value we want to match it to"""

    index_catalogs: dict[str, HCIndexCatalog]
    """Mapping of field name to respective index catalog"""

    def __init__(self, values: dict[str, Any], index_catalogs: dict[str, HCIndexCatalog], fine: bool = True):
        super().__init__(fine)
        if not all(key in index_catalogs for key in values):
            raise ValueError(
                f"There is a mismatch between the queried fields: "
                f"{values.keys()} and the fields of the provided index"
                f" catalogs: {index_catalogs.keys()}"
            )
        self.values = values
        self.index_catalogs = index_catalogs

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        """Determine the pixels for which there is a result in each field"""
        all_pixels = set(hc_structure.get_healpix_pixels())
        for field_name, field_value in self.values.items():
            field_value = field_value if isinstance(field_value, list) else [field_value]
            pixels_for_field = set(self.index_catalogs[field_name].loc_partitions(field_value))
            all_pixels = all_pixels.intersection(pixels_for_field)
        return hc_structure.filter_from_pixel_list(list(all_pixels))

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        filter_mask = np.ones(len(frame), dtype=np.bool)
        for field_name, field_index_catalog in self.index_catalogs.items():
            index_column = field_index_catalog.catalog_info.indexing_column
            field_values = (
                self.values[field_name]
                if isinstance(self.values[field_name], list)
                else [self.values[field_name]]
            )
            mask = frame[index_column].isin(field_values)
            filter_mask = filter_mask & mask
        return frame[filter_mask]
