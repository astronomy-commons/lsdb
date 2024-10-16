from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from upath import UPath

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.hats.abstract_catalog_loader import AbstractCatalogLoader, CatalogTypeVar
from lsdb.loaders.hats.association_catalog_loader import AssociationCatalogLoader
from lsdb.loaders.hats.hats_catalog_loader import HatsCatalogLoader
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.loaders.hats.margin_catalog_loader import MarginCatalogLoader

loader_class_for_catalog_type: Dict[Type[Dataset], Type[AbstractCatalogLoader]] = {
    Catalog: HatsCatalogLoader,
    AssociationCatalog: AssociationCatalogLoader,
    MarginCatalog: MarginCatalogLoader,
}


def get_loader_for_type(
    catalog_type_to_use: Type[CatalogTypeVar], path: str | Path | UPath, config: HatsLoadingConfig
) -> AbstractCatalogLoader:
    """Constructs a CatalogLoader that loads a Dataset of the specified type

    Args:
        catalog_type_to_use (Type[Dataset]): the type of catalog to be loaded. Uses the actual type
            as the input, not a string or enum value
        path (UPath): the path to load the catalog from
        config (HatsLoadingConfig): Additional configuration for loading the catalog

    Returns:
        An initialized CatalogLoader object with the path and config specified
    """
    if catalog_type_to_use not in loader_class_for_catalog_type:
        raise ValueError(f"Cannot load catalog type: {str(catalog_type_to_use)}")
    loader_class = loader_class_for_catalog_type[catalog_type_to_use]
    return loader_class(path, config)
