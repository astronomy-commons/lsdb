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
        if len(hc_catalog.get_healpix_pixels()) == 0:
            return None
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog)
