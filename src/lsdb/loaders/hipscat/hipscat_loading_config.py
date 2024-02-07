from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lsdb.catalog.margin_catalog import MarginCatalog


@dataclass
class HipscatLoadingConfig:
    """Configuration for loading a HiPSCat catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hipscat catalog.
    """

    columns: List[str] | None = None
    """Columns to load from the catalog - if not specified, all columns are loaded"""

    margin_cache: MarginCatalog | None = None

    kwargs: dict | None = None
    """Extra kwargs"""

    def get_kwargs_dict(self) -> dict:
        """Returns a dictionary with the extra kwargs"""
        return self.kwargs if self.kwargs is not None else {}
