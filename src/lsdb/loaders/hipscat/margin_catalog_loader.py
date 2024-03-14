import hipscat as hc

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class MarginCatalogLoader(AbstractCatalogLoader[MarginCatalog]):
    """Loads an HiPSCat MarginCatalog"""

    def load_catalog(self) -> MarginCatalog:
        hc_catalog = self._load_hipscat_catalog(hc.catalog.MarginCatalog)
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog)
