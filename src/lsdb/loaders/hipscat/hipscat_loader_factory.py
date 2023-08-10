from typing import TypeVar

from lsdb import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.hipscat_catalog_loader import HipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)


loader_class_for_catalog_type = {
    Catalog: HipscatCatalogLoader,
}


def get_loader_for_type(catalog_type_to_use: CatalogTypeVar, path: str, config: HipscatLoadingConfig):
    if catalog_type_to_use not in loader_class_for_catalog_type:
        raise ValueError(f"Cannot load catalog type: {str(catalog_type_to_use)}")
    LoaderClass = loader_class_for_catalog_type[catalog_type_to_use]
    return LoaderClass(path, config)
