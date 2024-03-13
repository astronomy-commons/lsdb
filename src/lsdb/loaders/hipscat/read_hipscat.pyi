"""Read Hipscat

Stub file containing typing for the read_hipscat function.

The read_hipscat method can either be used without specifying a catalog type, by default reading
from the catalog info the correct catalog type, however type checkers can't use the value from the
catalog info to infer the type. So there is also the option to specify the catalog type to ensure
correct type checking. This file specifies this typing of the function for the type checker to use.

For more information on stub files, view here: https://mypy.readthedocs.io/en/stable/stubs.html

"""

from typing import List, Type, overload

from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hipscat.abstract_catalog_loader import CatalogTypeVar

@overload
def read_hipscat(
    path: str,
    search_filter: AbstractSearch | None = None,
    storage_options: dict | None = None,
    columns: List[str] | None = None,
    margin_cache: MarginCatalog | None = None,
) -> Dataset: ...
@overload
def read_hipscat(
    path: str,
    catalog_type: Type[CatalogTypeVar],
    search_filter: AbstractSearch | None = None,
    storage_options: dict | None = None,
    columns: List[str] | None = None,
    margin_cache: MarginCatalog | None = None,
    **kwargs,
) -> CatalogTypeVar: ...
