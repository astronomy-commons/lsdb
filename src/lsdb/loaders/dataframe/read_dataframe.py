from __future__ import annotations

import re

from hipscat.io import FilePointer, get_file_pointer_from_path
from hipscat.io.file_io import get_basename_from_filepointer, is_regular_file, load_csv_to_pandas

from lsdb import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader


def read_dataframe(path: str, lowest_order: int = 0, threshold: int = 50, **kwargs) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        path (str): The path that locates the CSV file
        lowest_order (int): The lowest partition order
        threshold (int): The maximum number of data points per pixel
        **kwargs: Arguments to pass to the creation of the catalog info

    Returns:
        Catalog object loaded from the given parameters
    """
    file_pointer = get_file_pointer_from_path(path)
    _check_path_is_valid(file_pointer)
    df = load_csv_to_pandas(file_pointer)
    loader = DataframeCatalogLoader(df, lowest_order, threshold, **kwargs)
    return loader.load_catalog()


def _check_path_is_valid(path: FilePointer):
    """Checks if pointer to catalog file is valid. The pointer is
    valid if the file exists, and it is in CSV format.

    Args:
        path (FilePointer): Pointer to catalog file
    """
    if not is_regular_file(path):
        raise FileNotFoundError("Catalog file could not be found")
    file_name = get_basename_from_filepointer(path)
    match = re.match(r"(.*).csv", str(file_name))
    if not match:
        raise ValueError("Catalog file is not in CSV format")
