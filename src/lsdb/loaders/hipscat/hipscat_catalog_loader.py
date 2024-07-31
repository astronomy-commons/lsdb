from __future__ import annotations

import hipscat as hc

from lsdb.catalog.catalog import Catalog, MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig
from lsdb.loaders.hipscat.margin_catalog_loader import MarginCatalogLoader


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
        catalog = Catalog(dask_df, dask_df_pixel_map, filtered_hc_catalog)
        if self.config.search_filter is not None:
            catalog = catalog.search(self.config.search_filter)
        catalog.margin = self._load_margin_catalog()
        return catalog

    def _filter_hipscat_catalog(self, hc_catalog: hc.catalog.Catalog) -> hc.catalog.Catalog:
        """Filter the catalog pixels according to the spatial filter provided at loading time.
        Object and source catalogs are not allowed to be filtered to an empty catalog. If the
        resulting catalog is empty an error is issued indicating that the catalog does not have
        coverage for the desired region in the sky."""
        if self.config.search_filter is None:
            return hc_catalog
        filtered_catalog = self.config.search_filter.filter_hc_catalog(hc_catalog)
        if len(filtered_catalog.get_healpix_pixels()) == 0:
            raise ValueError("The selected sky region has no coverage")
        return hc.catalog.Catalog(
            filtered_catalog.catalog_info,
            filtered_catalog.pixel_tree,
            catalog_path=hc_catalog.catalog_path,
            moc=filtered_catalog.moc,
            schema=filtered_catalog.schema,
            storage_options=hc_catalog.storage_options,
        )

    def _load_margin_catalog(self) -> MarginCatalog | None:
        """Load the margin catalog. It can be provided using a margin catalog
        instance or a path to the catalog on disk."""
        margin_catalog = None
        if isinstance(self.config.margin_cache, MarginCatalog):
            margin_catalog = self.config.margin_cache
            if self.config.search_filter is not None:
                # pylint: disable=protected-access
                margin_catalog = margin_catalog.search(self.config.search_filter)
        elif self.config.margin_cache is not None:
            margin_catalog = MarginCatalogLoader(
                str(self.config.margin_cache),
                HipscatLoadingConfig(
                    search_filter=self.config.search_filter,
                    columns=self.config.columns,
                    margin_cache=None,
                    dtype_backend=self.config.dtype_backend,
                    **self.config.kwargs,
                ),
                storage_options=self.storage_options,
            ).load_catalog()
        return margin_catalog
