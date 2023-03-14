from typing import Dict, Type

from lsdb.catalog.catalog_source_type import CatalogSourceType
from lsdb.loaders.hipscat.abstract_hipscat_catalog_loader import \
    AbstractHipscatCatalogLoader
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig
from lsdb.loaders.hipscat.local_hipscat_catalog_loader import \
    LocalHipscatCatalogLoader

loaders: Dict[CatalogSourceType, Type[AbstractHipscatCatalogLoader]] = {
    CatalogSourceType.LOCAL: LocalHipscatCatalogLoader,
}


def build_hipscat_catalog_loader(
    path: str, source: CatalogSourceType, config: HipscatLoadingConfig
) -> AbstractHipscatCatalogLoader:
    """Generate a Catalog Loader object to load a catalog from a given source and parameters

    Args:
        source: The source to load the data from
        *args: Positional arguments to specify the location of the catalog in the source
        **kwargs: Keyword arguments to specify the location of the catalog in the source
    """
    if source not in loaders:
        raise ValueError(f"Catalog source type {source} is not supported")
    loader_class = loaders[source]
    return loader_class(path, config)
