from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.io._util import _arrow_dtype_mapping
from upath import UPath

from lsdb.core.search.abstract_search import AbstractSearch


@dataclass
class HatsLoadingConfig:
    """Configuration for loading a HATS catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hats-sharded catalog.
    """

    search_filter: AbstractSearch | None = None
    """The spatial filter to apply to the catalog"""

    columns: list[str] | None = None
    """Columns to load from the catalog. If not specified, all columns are loaded"""

    margin_cache: str | Path | UPath | None = None
    """Path to the margin cache catalog. Defaults to None."""

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
                # This is how HATS expects the filters to add to the url
                url_params["filters"].append(f"{filtr[0]}{filtr[1]}{filtr[2]}")

        return url_params

    def get_read_kwargs(self, kwargs_must_match_signature=None):
        """Clumps existing kwargs and `dtype_backend`, if specified.
        
        If `kwargs_must_match_signature` is specified, it will be used to filter through the kwargs
        and raise an error if any of the keys are not in the signature.
        """
        kwargs = dict(self.kwargs)

        if kwargs_must_match_signature:
            standard_params = ["engine", "filters"]

            signature = inspect.signature(kwargs_must_match_signature)
            signature_params = list(signature.parameters.keys())
            # I liked the theory of this, but the params are just:
            # ['func', 'iterables', 'args', 'meta', 'divisions', 'label', 'enforce_metadata', 'kwargs']
            # so it's not really useful. What kind of kwargs would we even be passing?

            for key in kwargs:
                if key not in standard_params and key not in signature_params:
                    raise ValueError(f"Key {key} is not a recognized parameter for type {kwargs_must_match_signature}") 

        if self.dtype_backend is not None:
            kwargs["dtype_backend"] = self.dtype_backend
        return kwargs
