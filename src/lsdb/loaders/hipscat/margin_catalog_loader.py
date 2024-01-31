import hipscat as hc

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class MarginCatalogLoader(AbstractCatalogLoader[MarginCatalog]):
    """Loads an HiPSCat MarginCatalog"""

    def load_catalog(self) -> MarginCatalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog)

    def load_hipscat_catalog(self) -> hc.catalog.MarginCatalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.MarginCatalog.read_from_hipscat(
            self.path, storage_options=self.storage_options
        )
