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

    required_fields: List[str] = dataclasses.field(default_factory=list)
    """Required parameters for loading the catalog"""

    def __post_init__(self):
        self._check_required_fields()

    def _check_required_fields(self):
        fields_dict = dataclasses.asdict(self)
        for field_name in self.required_fields:
            if field_name not in fields_dict or fields_dict[field_name] is None:
                raise ValueError(f"{field_name} is required to load the Catalog and a value must be provided")

    def as_dict(self) -> dict:
        """Returns the HipscatLoadingConfig as a dictionary"""
        config = dataclasses.asdict(self)
        del config["required_fields"]
        return config
