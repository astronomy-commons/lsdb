from __future__ import annotations

from pathlib import Path

import hats as hc
import nested_pandas as npd
import numpy as np
from fsspec.implementations.http import HTTPFileSystem
from hats.catalog import CatalogType
from hats.catalog.catalog_collection import CatalogCollection
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.io.file_io import file_io, get_upath
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from nested_pandas.nestedframe.io import from_pyarrow
from upath import UPath

import lsdb.nested as nd
from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog, DaskDFPixelMap, MarginCatalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.catalog.map_catalog import MapCatalog
from lsdb.catalog.margin_catalog import _validate_margin_catalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig

MAX_PYARROW_FILTERS = 10


def read_hats(
    path: str | Path | UPath,
    search_filter: AbstractSearch | None = None,
    columns: list[str] | str | None = None,
    margin_cache: str | Path | UPath | None = None,
    error_empty_filter: bool = True,
    **kwargs,
) -> Dataset:
    """Load catalog from a HATS path. See open_catalog()."""
    return open_catalog(path, search_filter, columns, margin_cache, error_empty_filter, **kwargs)


def open_catalog(
    path: str | Path | UPath,
    search_filter: AbstractSearch | None = None,
    columns: list[str] | str | None = None,
    margin_cache: str | Path | UPath | None = None,
    error_empty_filter: bool = True,
    filters=None,
    **kwargs,
) -> Dataset:
    """Open a catalog from a HATS path.

    Catalogs exist in collections or stand-alone.

    Catalogs in a HATS collection are composed of a main catalog, and margin and index
    catalogs. LSDB will open exactly ONE main object catalog and at most ONE margin catalog.
    The `collection.properties` file specifies which margins and indexes are available,
    and which margin to use by default::

        my_collection_dir/
        ├── main_catalog/
        ├── margin_catalog/
        ├── margin_catalog_2/
        ├── index_catalog/
        ├── collection.properties

    All arguments passed to the `open_catalog` call are applied to the calls to open
    the main and margin catalogs.

    Typical usage example, where we open a collection with a subset of columns::

        lsdb.open_catalog(path='./my_collection_dir', columns=['ra','dec'])

    Typical usage example, where we open a collection from a cone search::

        lsdb.open_catalog(
            path='./my_collection_dir',
            columns=['ra','dec'],
            search_filter=lsdb.ConeSearch(ra, dec, radius_arcsec),
        )

    Typical usage example, where we open a collection with a non-default margin::

        lsdb.open_catalog(path='./my_collection_dir', margin_cache='margin_catalog_2')

    Note that this margin still needs to be specified in the `all_margins` attribute
    of the `collection.properties` file.

    We can also open each catalog separately, if needed::

        lsdb.open_catalog(path='./my_collection_dir/main_catalog')

    Args:
        path (UPath | Path): The path that locates the root of the HATS collection or stand-alone catalog.
        search_filter (Type[AbstractSearch]): Default `None`. The filter method to be applied.
        columns (list[str] | str): Default `None`. The set of columns to filter the catalog on. If None,
            the catalog's default columns will be loaded. To load all catalog columns, use `columns="all"`.
        margin_cache (path-like): Default `None`. The margin for the main catalog, provided as a path.
        error_empty_filter (bool): Default True. If loading the catalog with a filter results in
            an empty catalog, throw error.
        filters (list[tuple[str]]): Default `None`. Filters to apply when reading parquet files.
            These may be applied as pyarrow filters or URL parameters.
        **kwargs: Arguments to pass to the pandas parquet file reader

    Returns:
        A `Catalog` object, and affiliated table links.

    Examples:
        To read a collection from a public S3 bucket, call it as follows::

            from upath import UPath
            collection = lsdb.open_catalog(UPath(..., anon=True))
    """
    hc_catalog = hc.read_hats(path)

    config = HatsLoadingConfig(
        search_filter=search_filter,
        columns=columns,
        error_empty_filter=error_empty_filter,
        margin_cache=margin_cache,
        filters=filters,
        kwargs=kwargs,
    )

    if isinstance(hc_catalog, CatalogCollection):
        config.margin_cache = _get_collection_margin(hc_catalog, margin_cache)
        catalog = _load_catalog(hc_catalog.main_catalog, config)
        catalog.hc_collection = hc_catalog  # type: ignore[attr-defined]
    else:
        catalog = _load_catalog(hc_catalog, config)
    return catalog


def _get_collection_margin(
    collection: CatalogCollection, margin_cache: str | Path | UPath | None
) -> UPath | None:
    """The path to the collection margin.

    The `margin_cache` should be provided as:
      - An identifier to the margin catalog name (it needs to be a string and be
        specified in the `all_margins` attribute of the `collection.properties`).
      - The absolute path to a margin, hosted locally or remote.

    By default, if no `margin_cache` is provided, the absolute path to the default
    collection margin is returned.
    """
    if margin_cache is None:
        return collection.default_margin_catalog_dir
    margin_cache = file_io.get_upath(margin_cache)
    if margin_cache.path in collection.all_margins:
        return collection.collection_path / margin_cache.path
    return margin_cache


def _load_catalog(hc_catalog: hc.catalog.Dataset, config: HatsLoadingConfig) -> Dataset:
    config.set_columns_from_catalog_info(hc_catalog.catalog_info)
    if hc_catalog.schema is None:
        raise ValueError(
            "The catalog schema could not be loaded from metadata."
            " Ensure your catalog has _common_metadata or _metadata files"
        )

    catalog_type = hc_catalog.catalog_info.catalog_type

    if catalog_type in (CatalogType.OBJECT, CatalogType.SOURCE):
        catalog = _load_object_catalog(hc_catalog, config)
    elif catalog_type == CatalogType.MARGIN:
        catalog = _load_margin_catalog(hc_catalog, config)
    elif catalog_type == CatalogType.ASSOCIATION:
        catalog = _load_association_catalog(hc_catalog, config)
    elif catalog_type == CatalogType.MAP:
        catalog = _load_map_catalog(hc_catalog, config)
    else:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")

    if (
        config.search_filter is not None
        and len(catalog.get_healpix_pixels()) == 0
        and config.error_empty_filter
    ):
        raise ValueError("The selected sky region has no coverage")

    catalog.hc_structure = _update_hc_structure(catalog)
    if isinstance(catalog, Catalog) and catalog.margin is not None:
        catalog.margin.hc_structure = _update_hc_structure(catalog.margin)
    return catalog


def _update_hc_structure(catalog: HealpixDataset):
    """Create the modified schema of the catalog after all the processing on the `read_hats` call"""
    # pylint: disable=protected-access
    return catalog._create_modified_hc_structure(
        updated_schema=get_arrow_schema(catalog._ddf),
        default_columns=(
            [col for col in catalog.hc_structure.catalog_info.default_columns if col in catalog._ddf.columns]
            if catalog.hc_structure.catalog_info.default_columns is not None
            else None
        ),
    )


def _load_association_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if hc_catalog.catalog_info.contains_leaf_files:
        dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    else:
        dask_meta_schema = _load_dask_meta_schema(hc_catalog, config)
        dask_df = nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)
        dask_df_pixel_map = {}
    return AssociationCatalog(dask_df, dask_df_pixel_map, hc_catalog, loading_config=config)


def _load_margin_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if config.search_filter:
        hc_catalog = config.search_filter.filter_hc_catalog(hc_catalog)
        pyarrow_filter = _generate_pyarrow_filters_from_moc(hc_catalog)
        if len(pyarrow_filter) > 0 and not config.filters:
            config.filters = pyarrow_filter
    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    margin = MarginCatalog(dask_df, dask_df_pixel_map, hc_catalog, loading_config=config)
    if config.search_filter is not None:
        margin = margin.search(config.search_filter)
    return margin


def _load_object_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    if config.search_filter:
        hc_catalog = config.search_filter.filter_hc_catalog(hc_catalog)
        if len(hc_catalog.get_healpix_pixels()) == 0 and config.error_empty_filter:
            raise ValueError("The selected sky region has no coverage")
        pyarrow_filter = _generate_pyarrow_filters_from_moc(hc_catalog)
        if len(pyarrow_filter) > 0 and not config.filters:
            config.filters = pyarrow_filter
    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    catalog = Catalog(dask_df, dask_df_pixel_map, hc_catalog, loading_config=config)
    if config.search_filter is not None:
        catalog = catalog.search(config.search_filter)
    if config.margin_cache is not None:
        margin_hc_catalog = hc.read_hats(config.margin_cache)
        margin = _load_margin_catalog(margin_hc_catalog, config)
        _validate_margin_catalog(margin_hc_catalog, hc_catalog)
        catalog.margin = margin
    return catalog


def _generate_pyarrow_filters_from_moc(filtered_catalog):
    pyarrow_filter = []
    if SPATIAL_INDEX_COLUMN not in filtered_catalog.schema.names:
        return pyarrow_filter
    if filtered_catalog.moc is not None:
        depth_array = filtered_catalog.moc.to_depth29_ranges
        if len(depth_array) > MAX_PYARROW_FILTERS:
            starts = depth_array.T[0]
            ends = depth_array.T[1]
            diffs = starts[1:] - ends[:-1]
            max_diff_inds = np.argpartition(diffs, -MAX_PYARROW_FILTERS)[-MAX_PYARROW_FILTERS:]
            max_diff_inds = np.sort(max_diff_inds)
            reduced_filters = []
            for i_start, i_end in zip(np.concat(([0], max_diff_inds)), np.concat((max_diff_inds, [-1]))):
                reduced_filters.append([starts[i_start], ends[i_end]])
            depth_array = np.array(reduced_filters)
        for hpx_range in depth_array:
            pyarrow_filter.append(
                [(SPATIAL_INDEX_COLUMN, ">=", hpx_range[0]), (SPATIAL_INDEX_COLUMN, "<", hpx_range[1])]
            )
    return pyarrow_filter


def _load_map_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    return MapCatalog(dask_df, dask_df_pixel_map, hc_catalog)


def _load_dask_meta_schema(hc_catalog, config) -> npd.NestedFrame:
    """Loads the Dask meta DataFrame from the parquet _metadata file."""
    columns = config.columns
    if (
        columns is not None
        and hc_catalog.schema is not None
        and SPATIAL_INDEX_COLUMN in hc_catalog.schema.names
        and SPATIAL_INDEX_COLUMN not in columns
    ):
        columns = columns + [SPATIAL_INDEX_COLUMN]
    dask_meta_schema = from_pyarrow(hc_catalog.schema.empty_table())
    if columns is not None:
        dask_meta_schema = dask_meta_schema[columns]
    if (
        dask_meta_schema.index.name != SPATIAL_INDEX_COLUMN
        and SPATIAL_INDEX_COLUMN in dask_meta_schema.columns
    ):
        dask_meta_schema = dask_meta_schema.set_index(SPATIAL_INDEX_COLUMN)
    if (
        config.columns is not None
        and SPATIAL_INDEX_COLUMN in config.columns
        and dask_meta_schema.index.name == SPATIAL_INDEX_COLUMN
    ):
        config.columns.remove(SPATIAL_INDEX_COLUMN)
    return dask_meta_schema


def _load_dask_df_and_map(catalog: HCHealpixDataset, config) -> tuple[nd.NestedFrame, DaskDFPixelMap]:
    """Load Dask DF from parquet files and make dict of HEALPix pixel to partition index"""
    pixels = catalog.get_healpix_pixels()
    ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
    divisions = get_pixels_divisions(ordered_pixels)
    dask_meta_schema = _load_dask_meta_schema(catalog, config)
    query_url_params = None
    if isinstance(get_upath(catalog.catalog_base_dir).fs, HTTPFileSystem):
        query_url_params = config.make_query_url_params()
    if len(ordered_pixels) > 0:
        ddf = nd.NestedFrame.from_map(
            read_pixel,
            ordered_pixels,
            catalog_base_dir=catalog.catalog_base_dir,
            npix_suffix=catalog.catalog_info.npix_suffix,
            query_url_params=query_url_params,
            columns=config.columns,
            schema=catalog.schema,
            filters=config.filters,
            **config.kwargs,
            divisions=divisions,
            meta=dask_meta_schema,
        )
    else:
        ddf = nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)
    pixel_to_index_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
    return ddf, pixel_to_index_map


def read_pixel(
    pixel: HealpixPixel,
    catalog_base_dir: str | Path | UPath,
    npix_suffix: str,
    *,
    query_url_params: dict | None = None,
    columns=None,
    schema=None,
    **kwargs,
):
    """Utility method to read a single pixel's parquet file from disk.

    NB: `columns` is necessary as an argument, even if None, so that dask-expr
    optimizes the execution plan."""

    return _read_parquet_file(
        hc.io.pixel_catalog_file(catalog_base_dir, pixel, query_url_params, npix_suffix=npix_suffix),
        columns=columns,
        schema=schema,
        **kwargs,
    )


def _read_parquet_file(
    path: UPath,
    *,
    columns=None,
    schema=None,
    **kwargs,
):
    if (
        columns is not None
        and schema is not None
        and SPATIAL_INDEX_COLUMN in schema.names
        and SPATIAL_INDEX_COLUMN not in columns
    ):
        columns = columns + [SPATIAL_INDEX_COLUMN]
    dataframe = file_io.read_parquet_file_to_pandas(path, columns=columns, schema=schema, **kwargs)

    if dataframe.index.name != SPATIAL_INDEX_COLUMN and SPATIAL_INDEX_COLUMN in dataframe.columns:
        dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN)

    return dataframe
