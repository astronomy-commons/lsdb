from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    error_empty_filter: bool = True
    """If loading raises an error for an empty filter result. Defaults to True."""

    kwargs: dict = field(default_factory=dict)
    """Extra kwargs for the pandas parquet file reader"""

    def __post_init__(self):
        # Check for commonly misspelled or mistaken keys
        for nonused_kwarg in ["margin", "maargin", "margins", "cache", "margincache"]:
            if nonused_kwarg in self.kwargs:
                raise ValueError(
                    f"Invalid keyword argument '{nonused_kwarg}' found. Did you mean 'margin_cache'?"
                )

    def make_query_url_params(self) -> dict:
        """
        Generates a dictionary of URL parameters with `columns` and `filters` attributes.
        """
        url_params: dict[str, Any] = {}

        if self.columns and len(self.columns) > 0:
            url_params["columns"] = self.columns

        if "filters" in self.kwargs:
            filters = []
            join_char = ","
            for filtr in self.kwargs["filters"]:
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

    def get_read_kwargs(self):
        """Clumps existing kwargs and `dtype_backend`, if specified."""
        kwargs = dict(self.kwargs)
        return kwargs
