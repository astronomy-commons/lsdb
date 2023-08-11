from __future__ import annotations

import dataclasses
from typing import Type, overload

from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar, get_loader_for_type
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


@overload
def read_hipscat(path: str) -> Catalog:
    ...


@overload
def read_hipscat(path: str, catalog_type: Type[CatalogTypeVar]) -> CatalogTypeVar:
    ...


def read_hipscat(
    path: str,
    catalog_type: Type[CatalogTypeVar] | None = None,
) -> CatalogTypeVar | Catalog:
    """Load a catalog from a HiPSCat formatted catalog.

    Args:
        path: The path that locates the root of the HiPSCat catalog
        catalog_type: Default `lsdb.Catalog` The type of catalog being loaded. Use by specifying the
             lsdb class for that catalog.
    Returns:
        Catalog object loaded from the given parameters
    """

    # Creates a config object to store loading parameters from all keyword arguments. I
    # originally had a few parameters in here, but after changing the file loading implementation
    # they weren't needed, so this object is now empty. But I wanted to keep this here for future
    # use
    kwd_args = locals().copy()
    config_args = {
        field.name: kwd_args[field.name]
        for field in dataclasses.fields(HipscatLoadingConfig)
    }
    config = HipscatLoadingConfig(**config_args)

    catalog_type_to_use: Type[Dataset] = Catalog

    if catalog_type is not None:
        catalog_type_to_use = catalog_type

    loader = get_loader_for_type(catalog_type_to_use, path, config)

    return loader.load_catalog()
