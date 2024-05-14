import dataclasses

import hipscat as hc

from lsdb.catalog.catalog import Catalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class HipscatCatalogLoader(AbstractCatalogLoader[Catalog]):
    """Loads a HiPSCat formatted Catalog"""

    def load_catalog(self) -> Catalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self._load_hipscat_catalog(hc.catalog.Catalog)
        search_filter = self.config.search_filter
        filtered_hc_catalog = (
            self._filter_hipscat_catalog(hc_catalog, search_filter)
            if search_filter is not None
            else hc_catalog
        )
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(filtered_hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, filtered_hc_catalog, self.config.margin_cache)

    def _filter_hipscat_catalog(
        self, hc_catalog: hc.catalog.Catalog, search_filter: AbstractSearch
    ) -> hc.catalog.Catalog:
        """Filters the catalog pixels according to the spatial filter provided at loading time"""
        pixels_to_load = search_filter.search_partitions(hc_catalog.get_healpix_pixels())
        if len(pixels_to_load) == 0:
            raise ValueError("The selected sky region has no coverage")
        catalog_info = dataclasses.replace(hc_catalog.catalog_info, total_rows=None)
        return hc.catalog.Catalog(
            catalog_info, pixels_to_load, self.path, hc_catalog.moc, self.storage_options
        )
