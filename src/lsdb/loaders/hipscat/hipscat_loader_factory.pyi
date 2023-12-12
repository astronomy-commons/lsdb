from typing import Dict, Type, TypeVar, Union, Any

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader, CatalogTypeVar
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig

loader_class_for_catalog_type: Dict[Type[Dataset], Type[AbstractCatalogLoader]]

def get_loader_for_type(
    catalog_type_to_use: Type[CatalogTypeVar],
    path: str,
    config: HipscatLoadingConfig,
    storage_options: Union[Dict[Any, Any], None] = None,
) -> AbstractCatalogLoader[CatalogTypeVar]: ...
