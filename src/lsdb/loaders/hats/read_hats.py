from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import hats as hc
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pyarrow as pa
from hats.catalog import CatalogType
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.io.file_io import file_io
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from upath import UPath

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog, DaskDFPixelMap, MarginCatalog
from lsdb.catalog.margin_catalog import _validate_margin_catalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.types import CatalogTypeVar


def read_hats(
    path: str | Path | UPath,
    search_filter: AbstractSearch | None = None,
    columns: List[str] | None = None,
    margin_cache: str | Path | UPath | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs,
) -> CatalogTypeVar | None:
    """Load a catalog from a HATS formatted catalog.

    Typical usage example, where we load a catalog with a subset of columns::

        lsdb.read_hats(path="./my_catalog_dir", columns=["ra","dec"])

    Typical usage example, where we load a catalog from a cone search::

        lsdb.read_hats(
            path="./my_catalog_dir",
            columns=["ra","dec"],
            search_filter=lsdb.core.search.ConeSearch(ra, dec, radius_arcsec),
        )

    Args:
        path (UPath | Path): The path that locates the root of the HATS catalog
        search_filter (Type[AbstractSearch]): Default `None`. The filter method to be applied.
        columns (List[str]): Default `None`. The set of columns to filter the catalog on.
        margin_cache (path-like): Default `None`. The margin for the main catalog, provided as a path.
        dtype_backend (str): Backend data type to apply to the catalog.
            Defaults to "pyarrow". If None, no type conversion is performed.
        **kwargs: Arguments to pass to the pandas parquet file reader

    Returns:
        Catalog object loaded from the given parameters
    """
    # Creates a config object to store loading parameters from all keyword arguments.
    config = HatsLoadingConfig(
        search_filter=search_filter,
        columns=columns,
        margin_cache=margin_cache,
        dtype_backend=dtype_backend,
        kwargs=kwargs,
    )

    hc_catalog = hc.read_hats(path)
    if hc_catalog.schema is None:
        raise ValueError(
            "The catalog schema could not be loaded from metadata."
            " Ensure your catalog has _common_metadata or _metadata files"
        )

    catalog_type = hc_catalog.catalog_info.catalog_type

    if catalog_type in (CatalogType.OBJECT, CatalogType.SOURCE):
        return _load_object_catalog(hc_catalog, config)
    if catalog_type == CatalogType.MARGIN:
        return _load_margin_catalog(hc_catalog, config)
    if catalog_type == CatalogType.ASSOCIATION:
        return _load_association_catalog(hc_catalog, config)

    raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")


def _load_association_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if hc_catalog.catalog_info.contains_leaf_files:
        dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    else:
        dask_meta_schema = _create_dask_meta_schema(hc_catalog.schema, config)
        dask_df = nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)
        dask_df_pixel_map = {}
    return AssociationCatalog(dask_df, dask_df_pixel_map, hc_catalog)


def _load_margin_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if config.search_filter:
        filtered_catalog = config.search_filter.filter_hc_catalog(hc_catalog)
        hc_catalog = hc.catalog.MarginCatalog(
            filtered_catalog.catalog_info,
            filtered_catalog.pixel_tree,
            catalog_path=hc_catalog.catalog_path,
            schema=filtered_catalog.schema,
            moc=filtered_catalog.moc,
        )
    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    margin = MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog)
    if config.search_filter is not None:
        margin = margin.search(config.search_filter)
    return margin


def _load_object_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if config.search_filter:
        filtered_catalog = config.search_filter.filter_hc_catalog(hc_catalog)
        if len(filtered_catalog.get_healpix_pixels()) == 0:
            raise ValueError("The selected sky region has no coverage")
        hc_catalog = hc.catalog.Catalog(
            filtered_catalog.catalog_info,
            filtered_catalog.pixel_tree,
            catalog_path=hc_catalog.catalog_path,
            moc=filtered_catalog.moc,
            schema=filtered_catalog.schema,
        )

    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    catalog = Catalog(dask_df, dask_df_pixel_map, hc_catalog)
    if config.search_filter is not None:
        catalog = catalog.search(config.search_filter)
    if config.margin_cache is not None:
        margin_hc_catalog = hc.read_hats(config.margin_cache)
        margin = _load_margin_catalog(margin_hc_catalog, config)
        _validate_margin_catalog(margin_hc_catalog, hc_catalog)
        catalog.margin = margin
    return catalog


def _create_dask_meta_schema(schema: pa.Schema, config) -> npd.NestedFrame:
    """Creates the Dask meta DataFrame from the HATS catalog schema."""
    dask_meta_schema = schema.empty_table().to_pandas(types_mapper=config.get_dtype_mapper())
    if (
        dask_meta_schema.index.name != SPATIAL_INDEX_COLUMN
        and SPATIAL_INDEX_COLUMN in dask_meta_schema.columns
    ):
        dask_meta_schema = dask_meta_schema.set_index(SPATIAL_INDEX_COLUMN)
        if config.columns is not None and SPATIAL_INDEX_COLUMN in config.columns:
            config.columns.remove(SPATIAL_INDEX_COLUMN)
    if config.columns is not None:
        dask_meta_schema = dask_meta_schema[config.columns]
    return npd.NestedFrame(dask_meta_schema)


def _load_dask_df_and_map(catalog: HCHealpixDataset, config) -> Tuple[nd.NestedFrame, DaskDFPixelMap]:
    """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
    pixels = catalog.get_healpix_pixels()
    ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
    divisions = get_pixels_divisions(ordered_pixels)
    dask_meta_schema = _create_dask_meta_schema(catalog.schema, config)
    if len(ordered_pixels) > 0:
        ddf = nd.NestedFrame.from_map(
            read_pixel,
            ordered_pixels,
            catalog=catalog,
            query_url_params=config.make_query_url_params(),
            columns=config.columns,
            divisions=divisions,
            meta=dask_meta_schema,
            schema=catalog.schema,
            **config.get_read_kwargs(),
        )
    else:
        ddf = nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)
    pixel_to_index_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
    return ddf, pixel_to_index_map


def read_pixel(
    pixel: HealpixPixel,
    catalog: HCHealpixDataset,
    *,
    query_url_params: dict | None = None,
    columns=None,
    schema=None,
    **kwargs,
):
    """Utility method to read a single pixel's parquet file from disk.

    NB: `columns` is necessary as an argument, even if None, so that dask-expr
    optimizes the execution plan."""
    if (
        columns is not None
        and schema is not None
        and SPATIAL_INDEX_COLUMN in schema.names
        and SPATIAL_INDEX_COLUMN not in columns
    ):
        columns = columns + [SPATIAL_INDEX_COLUMN]
    dataframe = file_io.read_parquet_file_to_pandas(
        hc.io.pixel_catalog_file(catalog.catalog_base_dir, pixel, query_url_params),
        columns=columns,
        schema=schema,
        **kwargs,
    )

    if dataframe.index.name != SPATIAL_INDEX_COLUMN and SPATIAL_INDEX_COLUMN in dataframe.columns:
        dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN)

    return dataframe
