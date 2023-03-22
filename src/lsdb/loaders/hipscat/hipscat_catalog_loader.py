import dask.dataframe as dd
import hipscat as hc
import pandas as pd
import pyarrow

from lsdb import io
from lsdb.catalog.catalog import Catalog, DaskDFPixelMap
from lsdb.core.healpix.healpix_pixel import MAXIMUM_ORDER, HealpixPixel
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


# pylint: disable=R0903
class HipscatCatalogLoader:
    """Loads a HiPSCat formatted Catalog from local files"""

    def __init__(self, path: str, config: HipscatLoadingConfig) -> None:
        """Initializes a LocalHipscatCatalogLoader

        Args:
            path: path to the root of the HiPSCat catalog
            config:
        """
        self.path = path
        self.base_catalog_dir = hc.io.get_file_pointer_from_path(self.path)
        self.config = config

    def load_catalog(self) -> Catalog:
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, hc_catalog)

    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.Catalog(catalog_path=self.path)

    def _load_dask_df_and_map(
        self, catalog: hc.catalog.Catalog
    ) -> tuple[dd.DataFrame, DaskDFPixelMap]:
        """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
        ordered_pixels = self._get_ordered_pixel_list(catalog)
        ordered_paths = self._get_paths_from_pixels(catalog, ordered_pixels)
        pixel_to_index_map = {
            pixel: index for index, pixel in enumerate(ordered_pixels)
        }
        ddf = self._load_df_from_paths(ordered_paths)
        return ddf, pixel_to_index_map

    def _get_ordered_pixel_list(
        self, catalog: hc.catalog.Catalog
    ) -> list[HealpixPixel]:
        pixels = []
        for _, row in catalog.get_pixels().iterrows():
            order = row[hc.catalog.PartitionInfo.METADATA_ORDER_COLUMN_NAME]
            pixel = row[hc.catalog.PartitionInfo.METADATA_PIXEL_COLUMN_NAME]
            pixels.append(HealpixPixel(order, pixel))
        # Sort pixels by pixel number at highest order
        sorted_pixels = sorted(
            pixels, key=lambda pixel: (4 ** (MAXIMUM_ORDER - pixel.order)) * pixel.pixel
        )
        return sorted_pixels

    def _get_paths_from_pixels(
        self, catalog: hc.catalog.Catalog, ordered_pixels: list[HealpixPixel]
    ) -> list[hc.io.FilePointer]:
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
        self, paths: list[hc.io.FilePointer]
    ) -> dd.DataFrame:
        metadata_schema = self._load_parquet_metadata_schema(paths)
        dask_meta_schema = self._get_schema_from_metadata(metadata_schema)
        ddf = dd.from_map(io.read_parquet_file_to_pandas, paths, meta=dask_meta_schema)
        return ddf

    def _load_parquet_metadata_schema(self, paths: list[hc.io.FilePointer]) -> pyarrow.Schema:
        return io.read_parquet_schema(
            paths[0]
        )

    def _get_schema_from_metadata(self, metadata_schema: pyarrow.Schema) -> pd.DataFrame:
        empty_table = metadata_schema.empty_table()
        return empty_table.to_pandas()
