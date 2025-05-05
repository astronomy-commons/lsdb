from __future__ import annotations

from pathlib import Path

import hats as hc
import nested_pandas as npd
import numpy as np
from hats.catalog import CatalogType
from hats.catalog.catalog_collection import CatalogCollection
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.io.file_io import file_io
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
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


def read_hats(
    path: str | Path | UPath,
    search_filter: AbstractSearch | None = None,
    columns: list[str] | str | None = None,
    margin_cache: str | Path | UPath | None = None,
    **kwargs,
) -> Dataset:
    """Load catalog from a HATS directory.

    Catalogs exist in collections or stand-alone.

    Catalogs in a HATS collection are composed of a main catalog, and margin and index
    catalogs. LSDB will load exactly ONE main object catalog, at most ONE margin catalog,
    and at most ONE index catalog. The `collection.properties` file specifies which
    margins and indexes are available, and which are the default::

        my_collection_dir/
        ├── main_catalog/
        ├── margin_catalog/
        ├── margin_catalog_2/
        ├── index_catalog/
        ├── collection.properties

    All arguments passed to the `read_hats` call are applied to the reading calls of
    the main and margin catalogs.

    Typical usage example, where we load a collection with a subset of columns::

        lsdb.read_hats(path='./my_collection_dir', columns=['ra','dec'])

    Typical usage example, where we load a collection from a cone search::

        lsdb.read_hats(
            path='./my_collection_dir',
            columns=['ra','dec'],
            search_filter=lsdb.core.search.ConeSearch(ra, dec, radius_arcsec),
        )

    Typical usage example, where we load a collection with a non-default margin::

        lsdb.read_hats(path='./my_collection_dir', margin_cache='margin_catalog_2')

    Note that this margin still needs to be specified in the `all_margins` attribute
    of the `collection.properties` file.

    We can also load each catalog separately, if needed::

        lsdb.read_hats(path='./my_collection_dir/main_catalog')

    Args:
        path (UPath | Path): The path that locates the root of the HATS collection or stand-alone catalog.
        search_filter (Type[AbstractSearch]): Default `None`. The filter method to be applied.
        columns (List[str]): Default `None`. The set of columns to filter the catalog on. If None, the
            catalog's default columns will be loaded. To load all catalog columns, use `columns="all"`
        margin_cache (path-like): Default `None`. The margin for the main catalog, provided as a path.
        dtype_backend (str): Backend data type to apply to the catalog.
            Defaults to "pyarrow". If None, no type conversion is performed.
        **kwargs: Arguments to pass to the pandas parquet file reader

    Returns:
        A `CatalogCollection` object if the provided path is for a HATS collection, a `Catalog` object
        if the path is for a stand-alone HATS catalog. Both are loaded from the given parameters.

    Examples:
        To read a collection from a public S3 bucket, call it as follows::

            from upath import UPath
            collection = lsdb.read_hats(UPath(..., anon=True))
    """
    hc_catalog = hc.read_hats(path)
    if isinstance(hc_catalog, CatalogCollection):
        margin_cache = _get_collection_margin(hc_catalog, margin_cache)
        catalog = _load_catalog(
            hc_catalog.main_catalog,
            search_filter=search_filter,
            columns=columns,
            margin_cache=margin_cache,
            **kwargs,
        )
        catalog.hc_collection = hc_catalog  # type: ignore[attr-defined]
    else:
        catalog = _load_catalog(
            hc_catalog,
            search_filter=search_filter,
            columns=columns,
            margin_cache=margin_cache,
            **kwargs,
        )
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


def _load_catalog(
    hc_catalog: hc.catalog.Dataset,
    search_filter: AbstractSearch | None = None,
    columns: list[str] | str | None = None,
    margin_cache: str | Path | UPath | None = None,
    **kwargs,
) -> Dataset:
    if columns is None and hc_catalog.catalog_info.default_columns is not None:
        columns = hc_catalog.catalog_info.default_columns

    if columns == "all":
        columns = None

    if isinstance(columns, str):
        raise TypeError("`columns` argument must be a list of strings, None, or 'all'")

    # Creates a config object to store loading parameters from all keyword arguments.
    config = HatsLoadingConfig(
        search_filter=search_filter,
        columns=columns,
        margin_cache=margin_cache,
        kwargs=kwargs,
    )

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

    if config.search_filter is not None and len(catalog.get_healpix_pixels()) == 0:
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


def _load_map_catalog(hc_catalog, config):
    """Load a catalog from the configuration specified when the loader was created

    Returns:
        Catalog object with data from the source given at loader initialization
    """
    dask_df, dask_df_pixel_map = _load_dask_df_and_map(hc_catalog, config)
    return MapCatalog(dask_df, dask_df_pixel_map, hc_catalog)


def _load_dask_meta_schema(hc_catalog, config) -> npd.NestedFrame:
    """Loads the Dask meta DataFrame from the parquet _metadata file."""
    metadata_pointer = hc.io.paths.get_common_metadata_pointer(hc_catalog.catalog_base_dir)
    dask_meta_schema = _read_parquet_file(metadata_pointer, columns=config.columns, schema=hc_catalog.schema)
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

    return _read_parquet_file(
        hc.io.pixel_catalog_file(
            catalog.catalog_base_dir, pixel, query_url_params, npix_suffix=catalog.catalog_info.npix_suffix
        ),
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
    dataframe = file_io.read_parquet_file_to_pandas(
        path,
        columns=columns,
        schema=schema,
        **kwargs,
    )

    if dataframe.index.name != SPATIAL_INDEX_COLUMN and SPATIAL_INDEX_COLUMN in dataframe.columns:
        dataframe = dataframe.set_index(SPATIAL_INDEX_COLUMN)

    return dataframe
