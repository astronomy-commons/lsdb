from __future__ import annotations

import dataclasses

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
        if filtered_hc_catalog is None:
            return None
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(filtered_hc_catalog)
        return MarginCatalog(dask_df, dask_df_pixel_map, filtered_hc_catalog)

    def _filter_hipscat_catalog(
        self, hc_catalog: hc.catalog.MarginCatalog
    ) -> hc.catalog.MarginCatalog | None:
        """Filter the catalog pixels according to the spatial filter provided at loading time.
        Margin catalogs, unlike object and source catalogs, are allowed to be filtered to an
        empty catalog. In that case, the margin catalog is considered None."""
        if self.config.search_filter is None:
            return hc_catalog
        pixels_to_load = self.config.search_filter.search_partitions(hc_catalog.get_healpix_pixels())
        if len(pixels_to_load) == 0:
            return None
        catalog_info = dataclasses.replace(hc_catalog.catalog_info, total_rows=None)
        return hc.catalog.MarginCatalog(
            catalog_info, pixels_to_load, self.path, hc_catalog.moc, self.storage_options
        )
