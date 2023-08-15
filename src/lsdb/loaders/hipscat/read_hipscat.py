from __future__ import annotations

import dataclasses
from typing import Type, overload, Dict

import hipscat as hc
from hipscat.catalog import CatalogType
from hipscat.catalog.dataset import BaseCatalogInfo

from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar, get_loader_for_type
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


dataset_class_for_catalog_type: Dict[CatalogType, Type[Dataset]] = {
    CatalogType.OBJECT: Catalog,
    CatalogType.SOURCE: Catalog,
}


def read_hipscat(
    path: str,
    catalog_type: Type[CatalogTypeVar] | None = None,
) -> CatalogTypeVar | Dataset:
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

    catalog_type_to_use = _get_dataset_class_from_catalog_info(path)

    if catalog_type is not None:
        catalog_type_to_use = catalog_type

    loader = get_loader_for_type(catalog_type_to_use, path, config)

    return loader.load_catalog()


def _get_dataset_class_from_catalog_info(base_catalog_path: str) -> Type[Dataset]:
    base_catalog_dir = hc.io.get_file_pointer_from_path(base_catalog_path)
    catalog_info_path = hc.io.paths.get_catalog_info_pointer(base_catalog_dir)
    catalog_info = BaseCatalogInfo.read_from_metadata_file(catalog_info_path)
    catalog_type = catalog_info.catalog_type
    if catalog_type not in dataset_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    return dataset_class_for_catalog_type[catalog_type]
