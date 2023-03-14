from typing import Tuple

import dask.dataframe as dd
import hipscat as hc
import hipscat.catalog
import pandas as pd
import pyarrow.parquet as pq

from lsdb.catalog.catalog import Catalog, DaskDFPixelMap
from lsdb.core.healpix.healpix_pixel import MAXIMUM_ORDER, HealpixPixel
from lsdb.loaders.hipscat.abstract_hipscat_catalog_loader import \
    AbstractHipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


# pylint: disable=R0903
class LocalHipscatCatalogLoader(AbstractHipscatCatalogLoader):
    """Loads a HiPSCat formatted Catalog from local files"""

    def __init__(self, path: str, config: HipscatLoadingConfig) -> None:
        """Initializes a LocalHipscatCatalogLoader

        Args:
            path: path to the root of the HiPSCat catalog
            config:
        """
        self.path = path
        self.config = config

    def load_catalog(self) -> Catalog:
        hc_catalog = self.load_hipscat_catalog()
        dask_df, dask_df_pixel_map = self._load_dask_df_and_map(hc_catalog)
        return Catalog(dask_df, dask_df_pixel_map, hc_catalog)

    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return hc.catalog.Catalog(catalog_path=self.path)

    def _load_dask_df_and_map(
        self, catalog: hipscat.catalog.Catalog
    ) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
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
    ) -> list[str]:
        paths = [
            hc.io.paths.pixel_catalog_file(
                catalog_path=catalog.catalog_path,
                pixel_order=pixel.order,
                pixel_number=pixel.pixel,
            )
            for pixel in ordered_pixels
        ]
        return paths

    def _load_df_from_paths(
        self, paths: list[str]
    ) -> dd.DataFrame:
        metadata_schema = self._load_parquet_metadata_schema(paths)
        dask_meta_schema = self._get_schema_from_metadata(metadata_schema)
        ddf = dd.from_map(pd.read_parquet, paths, meta=dask_meta_schema)
        return ddf

    def _load_parquet_metadata_schema(self, paths: list[str]):
        return pq.read_schema(
            paths[0]
        )

    def _get_schema_from_metadata(self, metadata_schema):
        index_column_field_name = self._get_index_column_from_metadata(metadata_schema)
        columns = {
            column["field_name"]: column
            for column in metadata_schema.pandas_metadata["columns"]
        }
        index_column = None
        if index_column_field_name is not None:
            index_column = pd.Series(
                dtype=columns[index_column_field_name]["pandas_type"]
            )
        df_columns = {
            column["name"]: pd.Series(dtype=column["pandas_type"])
            for field_name, column in columns.items()
            if field_name != index_column_field_name
        }
        schema_df = pd.DataFrame(df_columns, index=index_column)
        schema_df.index.name = columns[index_column_field_name]["name"]
        return schema_df

    def _get_index_column_from_metadata(self, metadata_schema) -> list[str]:
        index_col_name = None
        if "index_columns" in metadata_schema.pandas_metadata:
            index_cols = metadata_schema.pandas_metadata["index_columns"]
            if len(index_cols) > 1:
                raise NotImplementedError("Multiple index columns not supported")
            index_col_name = index_cols[0]
        return index_col_name
