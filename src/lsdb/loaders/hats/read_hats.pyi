"""Read HATS

Stub file containing typing for the read_hats function.

The read_hats method can either be used without specifying a catalog type, by default reading
from the catalog info the correct catalog type, however type checkers can't use the value from the
catalog info to infer the type. So there is also the option to specify the catalog type to ensure
correct type checking. This file specifies this typing of the function for the type checker to use.

For more information on stub files, view here: https://mypy.readthedocs.io/en/stable/stubs.html

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Type, overload

from upath import UPath

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hats.abstract_catalog_loader import CatalogTypeVar

@overload
def read_hats(
    path: str | Path | UPath,
    search_filter: AbstractSearch | None = None,
    columns: List[str] | None = None,
    margin_cache: str | Path | UPath | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs,
) -> Dataset | None: ...
@overload
def read_hats(
    path: str | Path | UPath,
    catalog_type: Type[CatalogTypeVar],
    search_filter: AbstractSearch | None = None,
    columns: List[str] | None = None,
    margin_cache: str | Path | UPath | None = None,
    dtype_backend: str | None = "pyarrow",
    **kwargs,
) -> CatalogTypeVar | None: ...
