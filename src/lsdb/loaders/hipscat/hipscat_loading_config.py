from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd
from hipscat.io.file_io import FilePointer
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

    margin_cache: MarginCatalog | FilePointer | None = None
    """Margin cache for the catalog. It can be provided as a path for the margin on disk,
    or as a margin object instance. By default, it is None."""

    dtype_backend: str | None = "pyarrow"
    """The backend data type to apply to the catalog. It defaults to "pyarrow" and 
    if it is None no type conversion is performed."""

    kwargs: dict = field(default_factory=dict)
    """Extra kwargs for the pandas parquet file reader"""

    def __post_init__(self):
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

    def make_query_url_params(self) -> dict:
        """
        Generates a dictionary of URL parameters with `columns` and `filters` attributes.
        """
        url_params = {}

        if self.columns and len(self.columns) > 0:
            url_params["columns"] = self.columns

        if "filters" in self.kwargs:
            url_params["filters"] = []
            for filtr in self.kwargs["filters"]:
                # This is how hipscat expects the filters to add to the url
                url_params["filters"].append(f"{filtr[0]}{filtr[1]}{filtr[2]}")

        return url_params
