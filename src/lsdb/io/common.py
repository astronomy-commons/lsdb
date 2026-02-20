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


def set_default_write_table_kwargs(write_table_kwargs):
    """Set common write table arguments.

    We set compression on parquet files to "ZSTD" level 15. In internal testing,
    we have found this to be most suitable for the kinds of data stored in
    Astronomy catalogs.

    Parameters
    ----------
    **kwargs :
        Arguments to pass to the parquet write operations

    Returns
    -------
    dict
        dictionary of keyword arguments to pass to parquet write operations.
    """
    if write_table_kwargs is None:
        write_table_kwargs = {}

    if "compression" not in write_table_kwargs:
        write_table_kwargs = write_table_kwargs | {
            "compression": "ZSTD",
            "compression_level": 15,
        }

    return write_table_kwargs
