from typing import List

import dask
import healpy as hp
import hipscat as hc
import pandas as pd


@dask.delayed
def polygon_filter(
    dataframe: pd.DataFrame, polygon_pixels: List[int], max_order: int, metadata: hc.catalog.Catalog
):
    """Filters a dataframe to only include points within the specified cone

    Args:
        dataframe (pd.DataFrame): DataFrame containing points in the sky
        polygon_pixels (List[int]): list of pixel indices at `max_order` within polygon
        max_order (int): order of `polygon_pixels`
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon
    """
    dataframe["_POLYGON_PIX"] = hp.ang2pix(
        2**max_order,
        dataframe[metadata.catalog_info.ra_column].values,
        dataframe[metadata.catalog_info.dec_column].values,
        lonlat=True,
        nest=True,
    )
    dataframe = dataframe.loc[dataframe["_POLYGON_PIX"].isin(polygon_pixels)]
    dataframe = dataframe.drop(columns=["_POLYGON_PIX"])
    return dataframe
