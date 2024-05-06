from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from pandas._libs import lib

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

    margin_cache: MarginCatalog | None = None
    """Margin cache for the catalog. By default, it is None"""

    dtype_backend: str = "pyarrow"
    """Whether the data should be backed by pyarrow or numpy. It is either 'pyarrow' or 'numpy'"""

    kwargs: dict | None = None
    """Extra kwargs"""

    def __post_init__(self):
        if self.dtype_backend not in ["pyarrow", "numpy"]:
            raise ValueError("The data type must be either 'pyarrow' or 'numpy'")

    def get_kwargs_dict(self) -> dict:
        """Returns a dictionary with the extra kwargs"""
        return self.kwargs if self.kwargs is not None else {}

    def get_dtype_backend(self) -> str:
        """Returns the data type backend. It is either "pyarrow" or <no_default>,
        in case we want to keep numpy-backed types."""
        return self.dtype_backend if self.dtype_backend == "pyarrow" else lib.no_default

    def get_pyarrow_dtype_mapper(self) -> pd.ArrowDtype | None:
        """Returns a types mapper for pyarrow"""
        return pd.ArrowDtype if self.dtype_backend == "pyarrow" else None
