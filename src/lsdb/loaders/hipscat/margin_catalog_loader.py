from __future__ import annotations

import hipscat as hc

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class MarginCatalogLoader(AbstractCatalogLoader[MarginCatalog]):
    """Loads an HiPSCat MarginCatalog"""

    def load_catalog(self) -> MarginCatalog | None:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self._load_hipscat_catalog(hc.catalog.MarginCatalog)
        filtered_hc_catalog = self._filter_hipscat_catalog(hc_catalog)
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(filtered_hc_catalog)
        margin = MarginCatalog(dask_df, dask_df_pixel_map, filtered_hc_catalog)
        if self.config.search_filter is not None:
            margin = margin.search(self.config.search_filter)
        return margin

    def _filter_hipscat_catalog(self, hc_catalog: hc.catalog.MarginCatalog) -> hc.catalog.MarginCatalog:
        """Filter the catalog pixels according to the spatial filter provided at loading time.
        Margin catalogs, unlike object and source catalogs, are allowed to be filtered to an
        empty catalog. In that case, the margin catalog is considered None."""
        if self.config.search_filter is None:
            return hc_catalog
        filtered_catalog = self.config.search_filter.filter_hc_catalog(hc_catalog)
        return hc.catalog.MarginCatalog(
            filtered_catalog.catalog_info,
            filtered_catalog.pixel_tree,
            catalog_path=hc_catalog.catalog_path,
            storage_options=hc_catalog.storage_options,
        )
