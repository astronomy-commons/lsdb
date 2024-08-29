from __future__ import annotations

from abc import abstractmethod
from typing import Generic, List, Tuple, Type

import hipscat as hc
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pyarrow as pa
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.io.file_io import file_io
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from upath import UPath

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
        self.base_catalog_dir = hc.io.file_io.get_upath(self.path)
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
        hc_catalog = catalog_type.read_from_hipscat(self.path, storage_options=self.storage_options)
        if hc_catalog.schema is None:
            raise ValueError(
                "The catalog schema could not be loaded from metadata."
                " Ensure your catalog has _common_metadata or _metadata files"
            )
        return hc_catalog

    def _load_dask_df_and_map(self, catalog: HCHealpixDataset) -> Tuple[nd.NestedFrame, DaskDFPixelMap]:
        """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
        pixels = catalog.get_healpix_pixels()
        ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
        # ordered_paths = self._get_paths_from_pixels(catalog, ordered_pixels)
        divisions = get_pixels_divisions(ordered_pixels)
        ddf = self._load_df_from_pixels(catalog, ordered_pixels, divisions)
        pixel_to_index_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
        return ddf, pixel_to_index_map

    def _get_paths_from_pixels(
        self, catalog: HCHealpixDataset, ordered_pixels: List[HealpixPixel]
    ) -> List[UPath]:
        paths = hc.io.paths.pixel_catalog_files(
            catalog.catalog_base_dir,
            ordered_pixels,
            self.config.make_query_url_params(),
            storage_options=self.storage_options,
        )
        return paths

    def _load_df_from_pixels(
        self, catalog: HCHealpixDataset, ordered_pixels: List[HealpixPixel], divisions: Tuple[int, ...] | None
    ) -> nd.NestedFrame:
        dask_meta_schema = self._create_dask_meta_schema(catalog.schema)
        if len(ordered_pixels) > 0:
            return nd.NestedFrame.from_map(
                read_pixel,
                ordered_pixels,
                catalog=catalog,
                query_url_params=self.config.make_query_url_params(),
                columns=self.config.columns,
                divisions=divisions,
                meta=dask_meta_schema,
                schema=catalog.schema,
                storage_options=self.storage_options,
                **self._get_kwargs(),
            )
        return nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)

    def _create_dask_meta_schema(self, schema: pa.Schema) -> npd.NestedFrame:
        """Creates the Dask meta DataFrame from the HiPSCat catalog schema."""
        dask_meta_schema = schema.empty_table().to_pandas(types_mapper=self.config.get_dtype_mapper())
        if self.config.columns is not None:
            dask_meta_schema = dask_meta_schema[self.config.columns]
        return npd.NestedFrame(dask_meta_schema)

    def _get_kwargs(self) -> dict:
        """Constructs additional arguments for the `read_parquet` call"""
        kwargs = dict(self.config.kwargs)
        if self.config.dtype_backend is not None:
            kwargs["dtype_backend"] = self.config.dtype_backend
        return kwargs


def read_pixel(
    pixel: HealpixPixel,
    catalog: HCHealpixDataset,
    query_url_params: str = None,
    **kwargs,
):
    return file_io.read_parquet_file_to_pandas(
        hc.io.pixel_catalog_file(
            catalog.catalog_base_dir,
            pixel,
            query_url_params,
        ),
        **kwargs,
    )
