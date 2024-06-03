from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd
from pandas.io._util import _arrow_dtype_mapping

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch


@dataclass
class HipscatLoadingConfig:
    """Configuration for loading a HiPSCat catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hipscat catalog.
    """

    search_filter: AbstractSearch | None = None
    """The spatial filter to apply to the catalog"""

    columns: List[str] | None = None
    """Columns to load from the catalog. If not specified, all columns are loaded"""

    margin_cache: MarginCatalog | str | None = None
    """Margin cache for the catalog. It can be provided as a path for the margin on disk,
    or as a margin object instance. By default, it is None."""

    dtype_backend: str | None = "pyarrow"
    """The backend data type to apply to the catalog. It defaults to "pyarrow" and 
    if it is None no type conversion is performed."""

    kwargs: dict = field(default_factory=dict)
    """Extra kwargs for the pandas parquet file reader"""

    def __post_init__(self):
        if self.margin_cache is not None and not isinstance(self.margin_cache, (MarginCatalog, str)):
            raise ValueError("`margin_cache` must be of type 'MarginCatalog' or 'str'")
        if self.dtype_backend not in ["pyarrow", "numpy_nullable", None]:
            raise ValueError("The data type backend must be either 'pyarrow' or 'numpy_nullable'")

    def get_dtype_mapper(self) -> Callable | None:
        """Returns a mapper for pyarrow or numpy types, mirroring Pandas behaviour."""
        mapper = None
        if self.dtype_backend == "pyarrow":
            mapper = pd.ArrowDtype
        elif self.dtype_backend == "numpy_nullable":
            mapper = _arrow_dtype_mapping().get
        return mapper
    
    def make_query_url_params(self) -> str:
        """Create a url query string from the search filter"""
        url_params = {}
        
        if self.columns and len(self.columns) > 0:
            url_params['columns'] = self.columns
        
        if "filters" in self.kwargs:
            url_params['filters'] = []
            for filter in self.kwargs["filters"]:
                # This is how lsdb server expects the filters, JSON doesnt support tuples
                url_params['filters'].append(f"{filter[0]}{filter[1]}{filter[2]}")
        
        # TODO: If kept "filters", this will raise a error when .compute() is called on the dask dataframe
        self.kwargs.pop("filters", None)
            
        return url_params
