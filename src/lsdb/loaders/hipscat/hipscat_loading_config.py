import dataclasses
from dataclasses import dataclass
from typing import List, Union


@dataclass
class HipscatLoadingConfig:
    """Configuration for loading a HiPSCat catalog in lsdb.

    Contains all parameters needed for a user to specify how to correctly read a hipscat catalog.
    """
    columns: Union[List[str],None] = None
    """Columns to load from the catalog - if not specified, all columns are loaded"""

    def as_dict(self) -> dict:
        """Returns the HipscatLoadingConfig as a dictionary"""
        return dataclasses.asdict(self)
