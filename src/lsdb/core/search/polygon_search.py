from typing import List

import dask
import healpy as hp
import hipscat as hc
import pandas as pd


@dask.delayed
def polygon_filter(
    data_frame: pd.DataFrame,
    polygon_pixels: List[int],
    max_order: int,
    metadata: hc.catalog.Catalog
):
    """Filters a dataframe to only include points within the specified polygon.

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        polygon_pixels (List[int]): list of pixel indices at `max_order` within polygon
        max_order (int): maximum order of `polygon_pixels`
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon.
    """
    data_frame["_POLYGON_PIX"] = hp.ang2pix(
        2**max_order,
        data_frame[metadata.catalog_info.ra_column].values,
        data_frame[metadata.catalog_info.dec_column].values,
        lonlat=True,
        nest=True,
    )
    data_frame = data_frame.loc[data_frame["_POLYGON_PIX"].isin(polygon_pixels)]
    data_frame = data_frame.drop(columns=["_POLYGON_PIX"])
    return data_frame
