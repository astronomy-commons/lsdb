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

dataset_class_for_catalog_type: Dict[CatalogType, Type[Dataset]] = {
    CatalogType.OBJECT: Catalog,
    CatalogType.SOURCE: Catalog,
    CatalogType.ASSOCIATION: AssociationCatalog,
    CatalogType.MARGIN: MarginCatalog,
}


# pylint: disable=unused-argument
def read_hats(
    path: str | Path | UPath,
    catalog_type: Type[CatalogTypeVar] | None = None,
    search_filter: AbstractSearch | None = None,
    columns: List[str] | None = None,
    margin_cache: MarginCatalog | str | Path | UPath | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs,
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
        columns (List[str]): Default `None`. The set of columns to filter the catalog on.
        margin_cache (MarginCatalog or path-like): The margin cache for the main catalog,
            provided as a path on disk or as an instance of the MarginCatalog object. Defaults to None.
        dtype_backend (str): Backend data type to apply to the catalog.
            Defaults to "pyarrow". If None, no type conversion is performed.
        **kwargs: Arguments to pass to the pandas parquet file reader

    Returns:
        Catalog object loaded from the given parameters
    """
    # Creates a config object to store loading parameters from all keyword arguments.
    kwd_args = locals().copy()
    config_args = {field.name: kwd_args[field.name] for field in dataclasses.fields(HatsLoadingConfig)}
    config = HatsLoadingConfig(**config_args)

    catalog_type_to_use = _get_dataset_class_from_catalog_info(path)

    if catalog_type is not None:
        catalog_type_to_use = catalog_type

    loader = get_loader_for_type(catalog_type_to_use, path, config)
    return loader.load_catalog()


def _get_dataset_class_from_catalog_info(base_catalog_path: str | Path | UPath) -> Type[Dataset]:
    base_catalog_dir = hc.io.file_io.get_upath(base_catalog_path)
    catalog_info = TableProperties.read_from_dir(base_catalog_dir)
    catalog_type = catalog_info.catalog_type
    if catalog_type not in dataset_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    return dataset_class_for_catalog_type[catalog_type]
