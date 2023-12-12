from typing import Any, Dict, Type, TypeVar, Union

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader
from lsdb.loaders.hipscat.association_catalog_loader import AssociationCatalogLoader
from lsdb.loaders.hipscat.hipscat_catalog_loader import HipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig

loader_class_for_catalog_type = {
    Catalog: HipscatCatalogLoader,
    AssociationCatalog: AssociationCatalogLoader,
}


def get_loader_for_type(
    catalog_type_to_use: Type[Dataset],
    path: str,
    config: HipscatLoadingConfig,
    storage_options: Union[Dict[Any, Any], None] = None,
) -> AbstractCatalogLoader:
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
    return loader_class(path, config, storage_options=storage_options)
