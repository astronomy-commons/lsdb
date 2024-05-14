from __future__ import annotations

import dataclasses

import hipscat as hc

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class MarginCatalogLoader(AbstractCatalogLoader[MarginCatalog]):
    """Loads an HiPSCat MarginCatalog"""

    def load_catalog(self) -> MarginCatalog | None:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self._load_hipscat_catalog(hc.catalog.MarginCatalog)
        search_filter = self.config.search_filter
        filtered_hc_catalog = (
            self._filter_hipscat_catalog(hc_catalog, search_filter)
            if search_filter is not None
            else hc_catalog
        )
        if filtered_hc_catalog is None:
            return None
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog)

    def _filter_hipscat_catalog(
        self, hc_catalog: hc.catalog.MarginCatalog, search_filter: AbstractSearch
    ) -> hc.catalog.MarginCatalog | None:
        """Filters the catalog pixels according to the spatial filter provided at loading time"""
        pixels_to_load = search_filter.search_partitions(hc_catalog.get_healpix_pixels())
        if len(pixels_to_load) == 0:
            return None
        catalog_info = dataclasses.replace(hc_catalog.catalog_info, total_rows=None)
        return hc.catalog.MarginCatalog(
            catalog_info, pixels_to_load, self.path, hc_catalog.moc, self.storage_options
        )
