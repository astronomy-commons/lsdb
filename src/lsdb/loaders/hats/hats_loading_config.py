from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from upath import UPath

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.loaders.hats.parquet_config import ParquetConfig


@dataclass
class HatsLoadingConfig:
    """Configuration for loading a HATS catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hats-sharded catalog.
    """

    search_filter: AbstractSearch | None = None
    """The spatial filter to apply to the catalog"""

    margin_cache: MarginCatalog | str | Path | UPath | None = None
    """Margin cache for the catalog. It can be provided as a path for the margin on disk,
    or as a margin object instance. By default, it is None."""

    parquet_config: ParquetConfig = field(default_factory=ParquetConfig)
    """Extra kwargs for the pandas parquet file reader."""

    @classmethod
    def create(cls, hats_kwargs, parquet_kwargs):
        """Creates a loader configuration from the `read_hats` kwargs."""
        return cls(**hats_kwargs, parquet_config=ParquetConfig(**parquet_kwargs))
