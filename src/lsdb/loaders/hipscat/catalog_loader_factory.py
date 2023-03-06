from typing import Dict, Type

from lsdb.catalog.catalog_source_type import CatalogSourceType
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader
from lsdb.loaders.hipscat.local_catalog_loader import LocalCatalogLoader

loaders: Dict[CatalogSourceType, Type[AbstractCatalogLoader]] = {
    CatalogSourceType.LOCAL: LocalCatalogLoader,
}


def build_catalog_loader_for_source(source: CatalogSourceType, *args, **kwargs):
    """Generate a Catalog Loader object to load a catalog from a given source and parameters

    Args:
        source: The source to load the data from
        *args: Positional arguments to specify the location of the catalog in the source
        **kwargs: Keyword arguments to specify the location of the catalog in the source
    """
    if source not in loaders:
        raise ValueError(f"Catalog source type {source} is not supported")
    loader_class = loaders[source]
    return loader_class(*args, **kwargs)
