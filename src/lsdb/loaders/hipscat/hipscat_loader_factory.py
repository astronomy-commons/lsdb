from __future__ import annotations

from typing import Dict, Type

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader, CatalogTypeVar
from lsdb.loaders.hipscat.association_catalog_loader import AssociationCatalogLoader
from lsdb.loaders.hipscat.hipscat_catalog_loader import HipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig
from lsdb.loaders.hipscat.margin_catalog_loader import MarginCatalogLoader

loader_class_for_catalog_type: Dict[Type[Dataset], Type[AbstractCatalogLoader]] = {
    Catalog: HipscatCatalogLoader,
    AssociationCatalog: AssociationCatalogLoader,
    MarginCatalog: MarginCatalogLoader,
}


def get_loader_for_type(
    catalog_type_to_use: Type[CatalogTypeVar],
    path: str,
    config: HipscatLoadingConfig,
    storage_options: dict | None = None,
) -> AbstractCatalogLoader:
    """Constructs a CatalogLoader that loads a Dataset of the specified type

    Args:
        catalog_type_to_use (Type[Dataset]): the type of catalog to be loaded. Uses the actual type
            as the input, not a string or enum value
        path (str): the path to load the catalog from
        config (HipscatLoadingConfig): Additional configuration for loading the catalog
        storage_options (dict): Dictionary that contains abstract filesystem credentials

    Returns:
        An initialized CatalogLoader object with the path and config specified
    """
    if catalog_type_to_use not in loader_class_for_catalog_type:
        raise ValueError(f"Cannot load catalog type: {str(catalog_type_to_use)}")
    loader_class = loader_class_for_catalog_type[catalog_type_to_use]
    return loader_class(path, config, storage_options=storage_options)
