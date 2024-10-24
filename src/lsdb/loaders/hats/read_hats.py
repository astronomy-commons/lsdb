from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Type

import hats as hc
from hats.catalog import CatalogType, TableProperties
from upath import UPath

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hats.abstract_catalog_loader import CatalogTypeVar
from lsdb.loaders.hats.hats_loader_factory import get_loader_for_type
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.loaders.hats.parquet_config import ParquetConfig

dataset_class_for_catalog_type: Dict[CatalogType, Type[Dataset]] = {
    CatalogType.OBJECT: Catalog,
    CatalogType.SOURCE: Catalog,
    CatalogType.ASSOCIATION: AssociationCatalog,
    CatalogType.MARGIN: MarginCatalog,
}


# pylint: disable=unused-argument
def read_hats(
    path: str | Path | UPath,
    *,
    catalog_type: Type[CatalogTypeVar] | None = None,
    search_filter: AbstractSearch | None = None,
    margin_cache: MarginCatalog | str | Path | UPath | None = None,
    columns: List[str] | None = None,
    filters: List[tuple] | List[list[tuple]] | None = None,
    dtype_backend: str | None = "pyarrow",
    **parquet_kwargs,
) -> CatalogTypeVar | None:
    """Load a catalog from a HATS formatted catalog.

    Typical usage example, where we load a catalog with a subset of columns::

        lsdb.read_hats(path="./my_catalog_dir", columns=["ra","dec"])

    Typical usage example, where we load a catalog from a cone search::

        lsdb.read_hats(
            path="./my_catalog_dir",
            catalog_type=lsdb.Catalog,
            columns=["ra","dec"],
            search_filter=lsdb.core.search.ConeSearch(ra, dec, radius_arcsec),
        )

    Args:
        path (UPath | Path): The path that locates the root of the HATS catalog
        catalog_type (Type[Dataset]): Default `None`. By default, the type of the catalog is loaded
            from the catalog info and the corresponding object type is returned. Python's type hints
            cannot allow a return type specified by a loaded value, so to use the correct return
            type for type checking, the type of the catalog can be specified here. Use by specifying
            the lsdb class for that catalog.
        search_filter (Type[AbstractSearch]): Default `None`. The filter method to be applied.
        margin_cache (MarginCatalog or path-like): The margin cache for the main catalog,
            provided as a path on disk or as an instance of the MarginCatalog object. Defaults to None.
        columns (List[str]): Default `None`. The set of columns to filter the catalog on.
        filters (List[tuple] | List[list[tuple]]): Filters to pass to the parquet reader.
        dtype_backend (str): Backend data type to apply to the catalog.
            Defaults to "pyarrow". If None, no type conversion is performed.
        **parquet_kwargs: Additional arguments to pass to the pandas parquet file reader.

    Returns:
        Catalog object loaded from the given parameters
    """
    # Creates a config object to store loading parameters from all keyword arguments.
    all_kwargs = locals().copy()
    hats_kwargs = _extract_hats_kwargs(**all_kwargs)
    parquet_kwargs = _extract_parquet_kwargs(**all_kwargs)
    config = HatsLoadingConfig.create(hats_kwargs, parquet_kwargs)
    catalog_type_to_use = _get_dataset_class_from_catalog_info(path)
    if catalog_type is not None:
        catalog_type_to_use = catalog_type
    loader = get_loader_for_type(catalog_type_to_use, path, config)
    return loader.load_catalog()


def _extract_hats_kwargs(**all_kwargs):
    """Create a dictionary with the HATS specific configuration fields"""
    hats_valid_fields = [
        field for field in dataclasses.fields(HatsLoadingConfig) if field.name != "parquet_config"
    ]
    return _extract_kwargs(hats_valid_fields, **all_kwargs)


def _extract_parquet_kwargs(**all_kwargs):
    """Create a dictionary with the parquet-reader specific configuration fields"""
    parquet_valid_fields = dataclasses.fields(ParquetConfig)
    explicit_kwargs = _extract_kwargs(parquet_valid_fields, **all_kwargs)
    implicit_kwargs = _extract_kwargs(parquet_valid_fields, **all_kwargs["parquet_kwargs"])
    return {**explicit_kwargs, **implicit_kwargs}


def _extract_kwargs(valid_fields: list, **kwargs):
    """Extract keyword arguments from a set of valid configuration fields"""
    return {field.name: kwargs[field.name] for field in valid_fields if field.name in kwargs}


def _get_dataset_class_from_catalog_info(base_catalog_path: str | Path | UPath) -> Type[Dataset]:
    base_catalog_dir = hc.io.file_io.get_upath(base_catalog_path)
    catalog_info = TableProperties.read_from_dir(base_catalog_dir)
    catalog_type = catalog_info.catalog_type
    if catalog_type not in dataset_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    return dataset_class_for_catalog_type[catalog_type]
