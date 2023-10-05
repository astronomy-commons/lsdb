from __future__ import annotations

import hipscat as hc
from hipscat.io import FilePointer, get_file_pointer_from_path

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar


def read_dataframe(
    path: str, catalog_name: str, ra_column: str = "ra", dec_column: str = "dec", threshold: int = 50
) -> CatalogTypeVar | Dataset:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        path (str): The path that locates the CSV file
        catalog_name (str): The name for the catalog
        ra_column (str): the name of the column for celestial coordinates, right ascension
        dec_column (str): the name of the column for celestial coordinates, declination
    Returns:
        Catalog object loaded from the given parameters
    """
    _check_path_is_valid(get_file_pointer_from_path(path))
    loader = DataframeCatalogLoader(
        path=path, threshold=threshold, catalog_name=catalog_name, ra_column=ra_column, dec_column=dec_column
    )
    return loader.load_catalog()


def _check_path_is_valid(path: FilePointer):
    """Checks if pointer to CSV file is valid."""
    file_exists = hc.io.file_io.is_regular_file(path)
    if not file_exists:
        raise FileNotFoundError("Catalog file could not be found")
