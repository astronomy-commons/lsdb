from __future__ import annotations

import pandas as pd

from lsdb.catalog import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader


def from_dataframe(
    dataframe: pd.DataFrame,
    lowest_order: int = 0,
    highest_order: int = 5,
    partition_size: int | None = None,
    threshold: int | None = None,
    **kwargs,
) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        dataframe (pd.Dataframe): The catalog Pandas Dataframe
        lowest_order (int): The lowest partition order
        highest_order (int): The highest partition order
        partition_size (int): The desired partition size, in number of rows
        threshold (int): The maximum number of data points per pixel
        **kwargs: Arguments to pass to the creation of the catalog info

    Returns:
        Catalog object loaded from the given parameters
    """
    loader = DataframeCatalogLoader(
        dataframe, lowest_order, highest_order, partition_size, threshold, **kwargs
    )
    return loader.load_catalog()
