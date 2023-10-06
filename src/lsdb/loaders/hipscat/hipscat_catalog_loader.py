from typing import List, Tuple, Union, Dict

import dask.dataframe as dd
import hipscat as hc
import pyarrow
from hipscat.io.file_io import file_io, file_pointer
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_HEALPIX_ORDER

from lsdb.catalog.catalog import Catalog, DaskDFPixelMap
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


class HipscatCatalogLoader:
    """Loads a HiPSCat formatted Catalog"""

    def __init__(
            self, path: str, config: HipscatLoadingConfig,
            storage_options: Union[Dict[str, str], None] = None) -> None:
        """Initializes a HipscatCatalogLoader

        Args:
            path: path to the root of the HiPSCat catalog
            config: options to configure how the catalog is loaded
        """
        self.path = path
        self.base_catalog_dir = hc.io.get_file_pointer_from_path(self.path)
        self.config = config
        self.storage_options = storage_options

    def load_catalog(self) -> Catalog:
        """Load a catalog from the configuration specified when the loader was created

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, hc_catalog)

    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.Catalog.read_from_hipscat(self.path, storage_options=self.storage_options)

    def _load_dask_df_and_map(self, catalog: hc.catalog.Catalog) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
        """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
        ordered_pixels = self._get_ordered_pixel_list(catalog)
        ordered_paths = self._get_paths_from_pixels(catalog, ordered_pixels)
        pixel_to_index_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
        ddf = self._load_df_from_paths(catalog, ordered_paths)
        return ddf, pixel_to_index_map

    def _get_ordered_pixel_list(self, catalog: hc.catalog.Catalog) -> List[HealpixPixel]:
        pixels = []
        for _, row in catalog.get_pixels().iterrows():
            order = row[hc.catalog.PartitionInfo.METADATA_ORDER_COLUMN_NAME]
            pixel = row[hc.catalog.PartitionInfo.METADATA_PIXEL_COLUMN_NAME]
            pixels.append(HealpixPixel(order, pixel))
        # Sort pixels by pixel number at highest order
        sorted_pixels = sorted(
            pixels,
            key=lambda pixel: (4 ** (HIPSCAT_ID_HEALPIX_ORDER - pixel.order)) * pixel.pixel,
        )
        return sorted_pixels

    def _get_paths_from_pixels(
        self, catalog: hc.catalog.Catalog, ordered_pixels: List[HealpixPixel]
    ) -> List[hc.io.FilePointer]:
        paths = [
            hc.io.paths.pixel_catalog_file(
                catalog_base_dir=catalog.catalog_base_dir,
                pixel_order=pixel.order,
                pixel_number=pixel.pixel,
            )
            for pixel in ordered_pixels
        ]
        return paths

    def _load_df_from_paths(
        self, catalog: hc.catalog.Catalog, paths: List[hc.io.FilePointer]
    ) -> dd.DataFrame:
        metadata_schema = self._load_parquet_metadata_schema(catalog, paths)
        dask_meta_schema = metadata_schema.empty_table().to_pandas()
        ddf = dd.from_map(
            file_io.read_parquet_file_to_pandas,
            paths,
            storage_options=self.storage_options,
            meta=dask_meta_schema
        )
        return ddf

    def _load_parquet_metadata_schema(
        self, catalog: hc.catalog.Catalog, paths: List[hc.io.FilePointer]
    ) -> pyarrow.Schema:
        metadata_pointer = hc.io.paths.get_parquet_metadata_pointer(catalog.catalog_base_dir)
        if file_pointer.does_file_or_directory_exist(metadata_pointer, storage_options=self.storage_options):
            metadata = file_io.read_parquet_metadata(metadata_pointer, storage_options=self.storage_options)
            return metadata.schema.to_arrow_schema()
        metadata = file_io.read_parquet_metadata(paths[0], storage_options=self.storage_options)
        return metadata.schema.to_arrow_schema()
