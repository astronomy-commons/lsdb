from __future__ import annotations

import dataclasses
from typing import Dict, Type

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
    storage_options: dict = None
) -> CatalogTypeVar | Dataset:
    """Load a catalog from a HiPSCat formatted catalog.

    Args:
        path (str): The path that locates the root of the HiPSCat catalog
        catalog_type (Type[Dataset]): Default `None`. By default, the type of the catalog is loaded
            from the catalog info and the corresponding object type is returned. Python's type hints
            cannot allow a return type specified by a loaded value, so to use the correct return
            type for type checking, the type of the catalog can be specified here. Use by specifying
            the lsdb class for that catalog.
    Returns:
        Catalog object loaded from the given parameters
    """

    # Creates a config object to store loading parameters from all keyword arguments. I
    # originally had a few parameters in here, but after changing the file loading implementation
    # they weren't needed, so this object is now empty. But I wanted to keep this here for future
    # use
    kwd_args = locals().copy()
    config_args = {field.name: kwd_args[field.name] for field in dataclasses.fields(HipscatLoadingConfig)}
    config = HipscatLoadingConfig(**config_args)

    catalog_type_to_use = _get_dataset_class_from_catalog_info(path, storage_options=storage_options)

    if catalog_type is not None:
        catalog_type_to_use = catalog_type

    loader = get_loader_for_type(catalog_type_to_use, path, config, storage_options=storage_options)
    return loader.load_catalog()


def _get_dataset_class_from_catalog_info(
        base_catalog_path: str, storage_options: dict = None
    ) -> Type[Dataset]:
    base_catalog_dir = hc.io.get_file_pointer_from_path(base_catalog_path)
    catalog_info_path = hc.io.paths.get_catalog_info_pointer(base_catalog_dir)
    print(catalog_info_path, storage_options)
    catalog_info = BaseCatalogInfo.read_from_metadata_file(catalog_info_path, storage_options=storage_options)
    catalog_type = catalog_info.catalog_type
    if catalog_type not in dataset_class_for_catalog_type:
        raise NotImplementedError(f"Cannot load catalog of type {catalog_type}")
    return dataset_class_for_catalog_type[catalog_type]
