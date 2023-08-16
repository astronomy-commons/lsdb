from typing import TypeVar, Type, Dict

from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.hipscat_catalog_loader import HipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)


loader_class_for_catalog_type: Dict[Type[Dataset], Type[HipscatCatalogLoader]] = {
    Catalog: HipscatCatalogLoader,
}


def get_loader_for_type(
        catalog_type_to_use: Type[CatalogTypeVar],
        path: str,
        config: HipscatLoadingConfig
) -> HipscatCatalogLoader:
    """Constructs a CatalogLoader that loads a Dataset of the specified type

    Args:
        catalog_type_to_use (Type[Dataset]): the type of catalog to be loaded. Uses the actual type
            as the input, not a string or enum value
        path (str): the path to load the catalog from
        config (HipscatLoadingConfig): Additional configuration for loading the catalog

    Returns:
        An initialized CatalogLoader object with the path and config specified
    """
    if catalog_type_to_use not in loader_class_for_catalog_type:
        raise ValueError(f"Cannot load catalog type: {str(catalog_type_to_use)}")
    loader_class = loader_class_for_catalog_type[catalog_type_to_use]
    return loader_class(path, config)
