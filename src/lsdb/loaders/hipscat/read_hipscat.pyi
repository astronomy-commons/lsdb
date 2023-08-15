from typing import overload, Type

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar


@overload
def read_hipscat(path: str) -> Dataset:
    ...


@overload
def read_hipscat(path: str, catalog_type: Type[CatalogTypeVar]) -> CatalogTypeVar:
    ...