from __future__ import annotations

from typing import TYPE_CHECKING

import nested_pandas as npd

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


class OrderSearch(AbstractSearch):
    """Filter the catalog by HEALPix order.

    Filters partitions in the catalog to those that are in the orders specified.
    Does not filter points inside those partitions.
    """

    def __init__(self, min_order: int = 0, max_order: int | None = None):
        super().__init__(fine=False)
        if max_order and min_order > max_order:
            raise ValueError("The minimum order should be lower than or equal to the maximum order")
        self.min_order = min_order
        self.max_order = max_order

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        max_catalog_order = hc_structure.pixel_tree.get_max_depth()
        max_order = max_catalog_order if self.max_order is None else self.max_order
        if self.min_order > max_order:
            raise ValueError("The minimum order is higher than the catalog's maximum order")
        pixels = [p for p in hc_structure.get_healpix_pixels() if self.min_order <= p.order <= max_order]
        return hc_structure.filter_from_pixel_list(pixels)

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        """Determine the search results within a data frame."""
        return frame
