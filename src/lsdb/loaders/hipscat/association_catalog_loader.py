import dask.dataframe as dd
import hipscat as hc

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class AssociationCatalogLoader(AbstractCatalogLoader[AssociationCatalog]):
    """Loads an HiPSCat AssociationCatalog"""

    def load_catalog(self) -> AssociationCatalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self._load_hipscat_catalog(hc.catalog.AssociationCatalog)
        if hc_catalog.catalog_info.contains_leaf_files:
            dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        else:
            dask_df, dask_df_pixel_map = self._load_empty_dask_df_and_map(hc_catalog)
        return AssociationCatalog(dask_df, dask_df_pixel_map, hc_catalog)

    def _load_empty_dask_df_and_map(self, hc_catalog):
        metadata_schema = self._load_parquet_metadata_schema(hc_catalog)
        dask_meta_schema = metadata_schema.empty_table().to_pandas()
        ddf = dd.io.from_pandas(dask_meta_schema, npartitions=0)
        return ddf, {}
