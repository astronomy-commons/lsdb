from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class HipscatLoadingConfig:
    """Configuration for loading a HiPSCat catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hipscat catalog.
    """
    columns: List[str] | None = None
    """Columns to load from the catalog - if not specified, all columns are loaded"""

    kwargs: dict | None = None
    """Extra kwargs"""
