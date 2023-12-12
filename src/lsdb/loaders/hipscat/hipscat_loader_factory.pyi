from typing import Any, Dict, Type, TypeVar, Union

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader, CatalogTypeVar
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig

def get_loader_for_type(
    catalog_type_to_use: Type[CatalogTypeVar],
    path: str,
    config: HipscatLoadingConfig,
    storage_options: Union[Dict[Any, Any], None] = None,
) -> AbstractCatalogLoader[CatalogTypeVar]: ...
