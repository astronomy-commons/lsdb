from __future__ import annotations

from hipscat.catalog import CatalogType

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader
from lsdb.loaders.hipscat.hipscat_loader_factory import CatalogTypeVar


def read_dataframe(
    path: str,
    catalog_name: str = "",
    catalog_type: CatalogType = None,
    ra_column: str = "ra",
    dec_column: str = "dec",
    threshold: int = 50,
) -> CatalogTypeVar | Dataset:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        path (str): The path that locates the CSV file
        catalog_name (str): The name for the catalog
        catalog_type (str): The type of catalog
        ra_column (str): the name of the column for celestial coordinates, right ascension
        dec_column (str): the name of the column for celestial coordinates, declination
        threshold (int): The maximum number of data points per pixel
    Returns:
        Catalog object loaded from the given parameters
    """
    loader = DataframeCatalogLoader(
        path=path,
        threshold=threshold,
        catalog_name=catalog_name,
        catalog_type=catalog_type,
        ra_column=ra_column,
        dec_column=dec_column,
    )
    return loader.load_catalog()
