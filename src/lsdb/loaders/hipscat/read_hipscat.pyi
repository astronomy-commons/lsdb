"""Read Hipscat

Stub file containing typing for the read_hipscat function.

The read_hipscat method can either be used without specifying a catalog type, by default reading
from the catalog info the correct catalog type, however type checkers can't use the value from the
catalog info to infer the type. So there is also the option to specify the catalog type to ensure
correct type checking. This file specifies this typing of the function for the type checker to use.

For more information on stub files, view here: https://mypy.readthedocs.io/en/stable/stubs.html

"""

from typing import Any, Dict, Type, Union, overload

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.hipscat.abstract_catalog_loader import CatalogTypeVar

@overload
def read_hipscat(path: str, storage_options: Union[Dict[Any, Any], None] = None) -> Dataset: ...
@overload
def read_hipscat(
    path: str, catalog_type: Type[CatalogTypeVar], storage_options: Union[Dict[Any, Any], None] = None
) -> CatalogTypeVar: ...
