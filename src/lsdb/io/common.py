from importlib.metadata import version
from pathlib import Path

from hats.catalog import TableProperties
from upath import UPath


def new_provenance_properties(path: str | Path | UPath | None = None, **kwargs) -> dict:
    """Create a new provenance properties dictionary for the dataset.

    Parameters
    ----------
    path: str | Path | UPath | None, default None
        The path to the catalog.
    **kwargs
        Additional provenance properties.

    Returns
    -------
    dict
        A new provenance dictionary.
    """
    return TableProperties.new_provenance_dict(path, builder=f"lsdb v{version('lsdb')}", **kwargs)
