from __future__ import annotations

import dataclasses

import hipscat as hc

import lsdb
from lsdb.catalog.catalog import Catalog, MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class HipscatCatalogLoader(AbstractCatalogLoader[Catalog]):
    """Loads a HiPSCat formatted Catalog"""

    def load_catalog(self) -> Catalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self._load_hipscat_catalog(hc.catalog.Catalog)
        filtered_hc_catalog = self._filter_hipscat_catalog(hc_catalog)
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(filtered_hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, filtered_hc_catalog, self._load_margin_catalog())

    def _filter_hipscat_catalog(self, hc_catalog: hc.catalog.Catalog) -> hc.catalog.Catalog:
        """Filter the catalog pixels according to the spatial filter provided at loading time.
        Object and source catalogs are not allowed to be filtered to an empty catalog. If the
        resulting catalog is empty an error is issued indicating that the catalog does not have
        coverage for the desired region in the sky."""
        if self.config.search_filter is None:
            return hc_catalog
        pixels_to_load = self.config.search_filter.search_partitions(hc_catalog.get_healpix_pixels())
        if len(pixels_to_load) == 0:
            raise ValueError("The selected sky region has no coverage")
        catalog_info = dataclasses.replace(hc_catalog.catalog_info, total_rows=None)
        return hc.catalog.Catalog(
            catalog_info, pixels_to_load, self.path, hc_catalog.moc, self.storage_options
        )

    def _load_margin_catalog(self) -> MarginCatalog | None:
        """Load the margin catalog. It can be provided using a margin catalog
        instance or a path to the catalog on disk."""
        margin_catalog = None
        if self.config.margin_cache is not None:
            margin_catalog = self.config.margin_cache
        elif self.config.margin_path is not None:
            margin_catalog = lsdb.read_hipscat(
                path=self.config.margin_path,
                catalog_type=MarginCatalog,
                search_filter=self.config.search_filter,
                dtype_backend=self.config.dtype_backend,
                storage_options=self.storage_options,
                **self.config.kwargs,
            )
        return margin_catalog
