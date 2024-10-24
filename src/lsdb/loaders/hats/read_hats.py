from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type

import hats as hc
from hats.catalog import CatalogType, TableProperties
from upath import UPath

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hats.abstract_catalog_loader import AbstractCatalogLoader, CatalogTypeVar
from lsdb.loaders.hats.association_catalog_loader import AssociationCatalogLoader
from lsdb.loaders.hats.hats_catalog_loader import HatsCatalogLoader
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.loaders.hats.margin_catalog_loader import MarginCatalogLoader

loader_class_for_catalog_type: Dict[CatalogType, Type[AbstractCatalogLoader]] = {
    CatalogType.OBJECT: HatsCatalogLoader,
    CatalogType.SOURCE: HatsCatalogLoader,
    CatalogType.ASSOCIATION: AssociationCatalogLoader,
    CatalogType.MARGIN: MarginCatalogLoader,
}


def read_hats(
    path: str | Path | UPath,
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
            columns=["ra","dec"],
            search_filter=lsdb.core.search.ConeSearch(ra, dec, radius_arcsec),
        )

    Args:
        path (UPath | Path): The path that locates the root of the HATS catalog
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
    # kwd_args = locals().copy()
    # config_args = {field.name: kwd_args[field.name] for field in dataclasses.fields(HatsLoadingConfig)}
    config = HatsLoadingConfig(
        search_filter=search_filter,
        columns=columns,
        margin_cache=margin_cache,
        dtype_backend=dtype_backend,
        kwargs=kwargs,
    )

    catalog_info = TableProperties.read_from_dir(hc.io.file_io.get_upath(path))
    catalog_type = catalog_info.catalog_type

    if catalog_type not in loader_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    loader_class = loader_class_for_catalog_type[catalog_type]
    return loader_class(path, config).load_catalog()
