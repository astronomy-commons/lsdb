from __future__ import annotations

import pandas as pd

from lsdb.catalog import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader


def from_dataframe(df: pd.DataFrame, lowest_order: int = 0, threshold: int = 100_000, **kwargs) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        df (pd.Dataframe): The catalog Pandas Dataframe
        lowest_order (int): The lowest partition order
        threshold (int): The maximum number of data points per pixel
        **kwargs: Arguments to pass to the creation of the catalog info

    Returns:
        Catalog object loaded from the given parameters
    """
    loader = DataframeCatalogLoader(df, lowest_order, threshold, **kwargs)
    return loader.load_catalog()
