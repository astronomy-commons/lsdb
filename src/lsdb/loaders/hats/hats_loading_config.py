from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
        url_params = {}

        if self.columns and len(self.columns) > 0:
            url_params["columns"] = self.columns

        if "filters" in self.kwargs:
            url_params["filters"] = []
            for filtr in self.kwargs["filters"]:
                # This is how HATS expects the filters to add to the url
                url_params["filters"].append(f"{filtr[0]}{filtr[1]}{filtr[2]}")

        return url_params

    def get_read_kwargs(self):
        """Clumps existing kwargs and `dtype_backend`, if specified."""
        kwargs = dict(self.kwargs)
        return kwargs
