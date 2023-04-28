from typing import Tuple

import hipscat as hc
import dask.dataframe as dd
import pyarrow
from hipscat.pixel_math import HealpixPixel

from lsdb import io
from lsdb.catalog.association_catalog.association_catalog import AssociationCatalog, \
    AssociationPixelMap
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


class HipscatAssociationCatalogLoader:
    """Loads a HiPSCat formatted Catalog"""

    def __init__(
            self, path: str, config: HipscatLoadingConfig
    ) -> None:
        """Initializes a HipscatCatalogLoader

        Args:
            path: path to the root of the HiPSCat catalog
            config: options to configure how the catalog is loaded
        """
        self.path = path
        self.base_catalog_dir = hc.io.get_file_pointer_from_path(self.path)
        self.config = config

    def load_catalog(self) -> AssociationCatalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return AssociationCatalog(dask_df, dask_df_pixel_map, hc_catalog)

    def load_hipscat_catalog(self) -> hc.catalog.AssociationCatalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.AssociationCatalog.read_from_hipscat(self.path)

    def _load_dask_df_and_map(
        self, catalog: hc.catalog.AssociationCatalog
    ) -> tuple[dd.core.DataFrame, AssociationPixelMap]:
        """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
        pixels = self._get_pixel_list(catalog)
        ordered_paths = self._get_paths_from_pixels(catalog, pixels)
        pixel_to_index_map = {
            pixel_tuple: index for index, pixel_tuple in enumerate(pixels)
        }
        ddf = self._load_df_from_paths(catalog, ordered_paths)
        return ddf, pixel_to_index_map

    def _get_pixel_list(
        self, catalog: hc.catalog.AssociationCatalog
    ) -> list[Tuple[HealpixPixel, HealpixPixel]]:
        pixels = []
        for _, row in catalog.get_join_pixels().iterrows():
            primary_order = row[hc.catalog.association_catalog.PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME]
            primary_pixel = row[hc.catalog.association_catalog.PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME]
            join_order = row[hc.catalog.association_catalog.PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME]
            join_pixel = row[hc.catalog.association_catalog.PartitionJoinInfo.JOIN_PIXEL_COLUMN_NAME]
            pixels.append((HealpixPixel(primary_order, primary_pixel), HealpixPixel(join_order, join_pixel)))
        return pixels

    def _get_paths_from_pixels(
        self, catalog: hc.catalog.AssociationCatalog, ordered_pixels: list[Tuple[HealpixPixel, HealpixPixel]]
    ) -> list[hc.io.FilePointer]:
        paths = [
            hc.io.paths.pixel_association_file(
                catalog_base_dir=catalog.catalog_base_dir,
                primary_pixel_order=primary_pixel.order,
                primary_pixel_number=primary_pixel.pixel,
                join_pixel_order=join_pixel.order,
                join_pixel_number=join_pixel.pixel,
            )
            for primary_pixel, join_pixel in ordered_pixels
        ]
        return paths

    def _load_df_from_paths(
        self, catalog: hc.catalog.AssociationCatalog, paths: list[hc.io.FilePointer]
    ) -> dd.core.DataFrame:
        metadata_schema = self._load_parquet_metadata_schema(catalog, paths)
        dask_meta_schema = metadata_schema.empty_table().to_pandas().set_index("primary_hipscat_index")
        ddf = dd.from_map(io.read_parquet_file_to_pandas, paths, meta=dask_meta_schema)
        return ddf

    def _load_parquet_metadata_schema(
        self, catalog: hc.catalog.AssociationCatalog, paths: list[hc.io.FilePointer]
    ) -> pyarrow.Schema:
        metadata_pointer = hc.io.paths.get_parquet_metadata_pointer(
            catalog.catalog_base_dir
        )
        if hc.io.file_io.does_file_or_directory_exist(metadata_pointer):
            return io.read_parquet_schema(metadata_pointer)
        return io.read_parquet_schema(paths[0])
