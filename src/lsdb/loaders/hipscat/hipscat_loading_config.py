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

    use_pyarrow_types: bool = True
    """Whether the data should be backed by pyarrow or not. Defaults to "pyarrow"."""

    kwargs: dict | None = None
    """Extra kwargs"""

    def get_kwargs_dict(self) -> dict:
        """Returns a dictionary with the extra kwargs"""
        return self.kwargs if self.kwargs is not None else {}

    def get_dtype_backend(self) -> str:
        """Returns the data type backend. It is either "pyarrow" or <no_default>,
        in case we want to keep the original types."""
        return "pyarrow" if self.use_pyarrow_types else lib.no_default

    def get_pyarrow_dtype_mapper(self) -> pd.ArrowDtype | None:
        """Returns a mapper for pyarrow types"""
        return pd.ArrowDtype if self.use_pyarrow_types else None
