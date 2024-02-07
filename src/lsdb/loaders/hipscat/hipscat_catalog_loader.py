import hipscat as hc

from lsdb.catalog.catalog import Catalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class HipscatCatalogLoader(AbstractCatalogLoader[Catalog]):
    """Loads a HiPSCat formatted Catalog"""

    def load_catalog(self) -> Catalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, hc_catalog, self.config.margin_cache)

    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.Catalog.read_from_hipscat(self.path, storage_options=self.storage_options)
