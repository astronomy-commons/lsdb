from __future__ import annotations

import math

import pandas as pd

from lsdb.catalog import Catalog
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader


def from_dataframe(
        df: pd.DataFrame,
        lowest_order: int = 0,
        highest_order: int = 10,
        partition_size: float | None = None,
        threshold: int | None = None,
        **kwargs,
) -> Catalog:
    """Load a catalog from a Pandas Dataframe in CSV format.

    Args:
        df (pd.Dataframe): The catalog Pandas Dataframe
        lowest_order (int): The lowest partition order
        highest_order (int): The highest partition order
        partition_size (float): The desired partition size, in megabytes
        threshold (int): The maximum number of data points per pixel
        **kwargs: Arguments to pass to the creation of the catalog info

    Returns:
        Catalog object loaded from the given parameters
    """
    if threshold is not None and partition_size is not None:
        raise ValueError("Specify only one: threshold or partition_size")
    if threshold is None and partition_size is None:
        threshold = DataframeCatalogLoader.DEFAULT_THRESHOLD
    elif threshold is None and partition_size is not None:
        threshold = calculate_threshold(df, partition_size)
    assert isinstance(threshold, int)
    loader = DataframeCatalogLoader(df, lowest_order, highest_order, threshold, **kwargs)
    return loader.load_catalog()


def calculate_threshold(df: pd.Dataframe, partition_size: float = 100):
    """Calculates the number of pixels per HEALPix pixel (threshold)
    for the desired partition size.

    Args:
        df (pd.Dataframe): The catalog Pandas Dataframe
        partition_size (float): The desired partition size, in megabytes

    Returns:
        The HEALPix pixel threshold
    """
    df_size_bytes = df.memory_usage().sum()
    # Round the number of partitions to the next integer, otherwise the
    # number of pixels per partition may exceed the threshold
    num_partitions = math.ceil(df_size_bytes / (partition_size * (1 << 20)))
    return len(df.index) // num_partitions
