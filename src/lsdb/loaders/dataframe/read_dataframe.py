from __future__ import annotations

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar


def read_dataframe(
    path: str, catalog_name: str, ra_column: str = "ra", dec_column: str = "dec"
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
    loader = DataframeCatalogLoader(
        path=path, catalog_name=catalog_name, ra_column=ra_column, dec_column=dec_column
    )
    return loader.load_catalog()
