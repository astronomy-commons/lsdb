from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas.io._util import _arrow_dtype_mapping


@dataclass
class ParquetConfig:
    """Parquet-reader specific configuration parameters"""

    engine: str = "pyarrow"
    """Parquet library to use. Defaults to 'pyarrow'"""

    columns: list[str] | None = None
    """Columns to load from the catalog. If not specified, all columns are loaded"""

    dtype_backend: str | None = None
    """The backend data type to apply to the catalog. One of 'numpy_nullable' or 'pyarrow'. 
    Defaults to None, where no conversion is performed."""

    filters: list[tuple] | list[list[tuple]] | None = None
    """Expressions to filter data on. Filter syntax: [[(column, op, val), …],…] 
    where op is [==, =, >, >=, <, <=, !=, in, not in]."""

    def __post_init__(self):
        if self.dtype_backend not in ["pyarrow", "numpy_nullable", None]:
            raise ValueError("The data type backend must be either 'pyarrow' or 'numpy_nullable'")

    def get_dtype_mapper(self) -> Callable | None:
        """Return a mapper for pyarrow or numpy types, mirroring Pandas behaviour."""
        mapper = None
        if self.dtype_backend == "pyarrow":
            mapper = pd.ArrowDtype
        elif self.dtype_backend == "numpy_nullable":
            mapper = _arrow_dtype_mapping().get
        return mapper

    def make_query_url_params(self) -> dict:
        """Generate a dictionary of URL parameters with `columns` and `filters` attributes."""
        url_params = {}
        if self.columns and len(self.columns) > 0:
            url_params["columns"] = self.columns
        if self.filters and len(self.filters) > 0:
            url_params["filters"] = []
            for filtr in self.filters:
                # This is how HATS expects the filters to add to the url
                url_params["filters"].append(f"{filtr[0]}{filtr[1]}{filtr[2]}")
        return url_params

    def generate_kwargs(self) -> dict:
        """Generate the dictionary of keyword arguments."""
        kwargs = dataclasses.asdict(self)
        if self.dtype_backend is None:
            del kwargs["dtype_backend"]
        return kwargs
