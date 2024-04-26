from __future__ import annotations

from abc import abstractmethod
from typing import Generic, List, Tuple, Type

import dask.dataframe as dd
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.io.file_io import file_io
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from lsdb.catalog.catalog import DaskDFPixelMap
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig
from lsdb.types import CatalogTypeVar, HCCatalogTypeVar


class AbstractCatalogLoader(Generic[CatalogTypeVar]):
    """Loads a HiPSCat Dataset with the type specified by the type variable"""

    def __init__(self, path: str, config: HipscatLoadingConfig, storage_options: dict | None = None) -> None:
        """Initializes a HipscatCatalogLoader

        Args:
            path: path to the root of the HiPSCat catalog
            config: options to configure how the catalog is loaded
            storage_options: options for the file system the catalog is loaded from
        """
        self.path = path
        self.base_catalog_dir = hc.io.get_file_pointer_from_path(self.path)
        self.config = config
        self.storage_options = storage_options

    @abstractmethod
    def load_catalog(self) -> CatalogTypeVar | None:
        """Load a dataset from the configuration specified when the loader was created

        Returns:
            Dataset object of the class's type with data from the source given at loader initialization
        """
        pass

    def _load_hipscat_catalog(self, catalog_type: Type[HCCatalogTypeVar]) -> HCCatalogTypeVar:
        """Load `hipscat` library catalog object with catalog metadata and partition data"""
        return catalog_type.read_from_hipscat(self.path, storage_options=self.storage_options)

    def _load_dask_df_and_map(self, catalog: HCHealpixDataset) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
        """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
        pixels = catalog.get_healpix_pixels()
        ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
        ordered_paths = self._get_paths_from_pixels(catalog, ordered_pixels)
        divisions = get_pixels_divisions(ordered_pixels)
        ddf = self._load_df_from_paths(catalog, ordered_paths, divisions)
        pixel_to_index_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
        return ddf, pixel_to_index_map

    def _get_paths_from_pixels(
        self, catalog: HCHealpixDataset, ordered_pixels: List[HealpixPixel]
    ) -> List[hc.io.FilePointer]:
        paths = hc.io.paths.pixel_catalog_files(
            catalog.catalog_base_dir,
            ordered_pixels,
            self.config.make_query_url_params(),
            storage_options=self.storage_options,
        )
        return paths

    def _load_df_from_paths(
        self, catalog: HCHealpixDataset, paths: List[hc.io.FilePointer], divisions: Tuple[int, ...] | None
    ) -> dd.DataFrame:
        dask_meta_schema = self._load_metadata_schema(catalog)
        if self.config.columns:
            dask_meta_schema = dask_meta_schema[self.config.columns]
        kwargs = dict(self.config.kwargs)
        if self.config.dtype_backend is not None:
            kwargs["dtype_backend"] = self.config.dtype_backend
        if len(paths) > 0:
            return dd.from_map(
                file_io.read_parquet_file_to_pandas,
                paths,
                columns=self.config.columns,
                divisions=divisions,
                meta=dask_meta_schema,
                storage_options=self.storage_options,
                **kwargs,
            )
        return dd.from_pandas(dask_meta_schema, npartitions=1)

    def _load_metadata_schema(self, catalog: HCHealpixDataset) -> pd.DataFrame:
        metadata_pointer = hc.io.paths.get_common_metadata_pointer(catalog.catalog_base_dir)
        metadata = file_io.read_parquet_metadata(metadata_pointer, storage_options=self.storage_options)
        return (
            metadata.schema.to_arrow_schema()
            .empty_table()
            .to_pandas(types_mapper=self.config.get_dtype_mapper())
        )
