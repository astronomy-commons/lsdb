from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from upath import UPath

from lsdb.core.search.abstract_search import AbstractSearch


@dataclass
class HatsLoadingConfig:
    """Configuration for loading a HATS catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hats-sharded catalog.
    """

    search_filter: AbstractSearch | None = None
    """The spatial filter to apply to the catalog"""

    columns: list[str] | str | None = None
    """Columns to load from the catalog. If not specified, all columns are loaded"""

    margin_cache: str | Path | UPath | None = None
    """Path to the margin cache catalog. Defaults to None."""

    error_empty_filter: bool = True
    """If loading raises an error for an empty filter result. Defaults to True."""

    filters: Any = None
    """Pyarrow filters to apply on reading parquet"""

    kwargs: dict = field(default_factory=dict)
    """Extra kwargs for the pandas parquet file reader"""

    def __post_init__(self):
        # Check for commonly misspelled or mistaken keys
        for nonused_kwarg in ["margin", "maargin", "margins", "cache", "margincache"]:
            if nonused_kwarg in self.kwargs:
                raise ValueError(
                    f"Invalid keyword argument '{nonused_kwarg}' found. Did you mean 'margin_cache'?"
                )

    def set_columns_from_catalog_info(self, catalog_info):
        """Set the appropriate columns to load, based on the user-provided `columns` argument
        and the actual columns of the dataset."""
        columns = self.columns
        if columns is None and catalog_info.default_columns is not None:
            columns = catalog_info.default_columns
        if isinstance(columns, str):
            if columns != "all":
                raise TypeError("`columns` argument must be a sequence of strings, None, or 'all'")
            columns = None
        elif pd.api.types.is_list_like(columns):
            columns = list(columns)  # type: ignore[arg-type]

        ra_col = catalog_info.ra_column
        dec_col = catalog_info.dec_column
        if columns is not None:
            if ra_col is not None and ra_col not in columns:
                columns.append(ra_col)
            if dec_col is not None and dec_col not in columns:
                columns.append(dec_col)
        self.columns = columns

    def make_query_url_params(self) -> dict:
        """
        Generates a dictionary of URL parameters with `columns` and `filters` attributes.
        """
        url_params: dict[str, Any] = {}

        if self.columns and len(self.columns) > 0:
            url_params["columns"] = self.columns

        if self.filters:
            filters = []
            join_char = ","
            for filtr in self.filters:
                # This is how HATS expects the filters to add to the url, supporting the list forms matching
                # pyarrow https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
                if isinstance(filtr[0], str):
                    # If list of filter tuples, generate comma seperated list of filters to get a conjunction
                    # the overall condition will be the 'and' of all the filters
                    filters.append(f"{filtr[0]}{filtr[1]}{filtr[2]}")
                else:
                    # If nested list of filter tuples, generate comma seperated list of filters to get a
                    # conjunction (AND) for the inner list of filters, and join those lists with ';' to make a
                    # disjunction (OR) of the inner conjunctions.
                    join_char = ";"
                    conj = []
                    for f in filtr:
                        conj.append(f"{f[0]}{f[1]}{f[2]}")
                    filters.append(",".join(conj))
            url_params["filters"] = join_char.join(filters)

        return url_params
