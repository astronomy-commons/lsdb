from __future__ import annotations

import pandas as pd

from lsdb.catalog import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader
from lsdb.loaders.dataframe.margin_catalog_generator import MarginCatalogGenerator


def from_dataframe(
    dataframe: pd.DataFrame,
    lowest_order: int = 0,
    highest_order: int = 5,
    partition_size: int | None = None,
    threshold: int | None = None,
    margin_order: int | None = -1,
    margin_threshold: float = 5.0,
    **kwargs,
) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        dataframe (pd.Dataframe): The catalog Pandas Dataframe
        lowest_order (int): The lowest partition order
        highest_order (int): The highest partition order
        partition_size (int): The desired partition size, in number of rows
        threshold (int): The maximum number of data points per pixel
        margin_order (int): The order at which to generate the margin cache
        margin_threshold (float): The size of the margin cache boundary, in arcseconds
        **kwargs: Arguments to pass to the creation of the catalog info

    Returns:
        Catalog object loaded from the given parameters
    """
    catalog = DataframeCatalogLoader(
        dataframe,
        lowest_order,
        highest_order,
        partition_size,
        threshold,
        **kwargs,
    ).load_catalog()
    if margin_threshold:
        catalog.margin = MarginCatalogGenerator(
            catalog,
            margin_order,
            margin_threshold,
        ).create_catalog()
    return catalog
