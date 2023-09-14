import dask
import hipscat as hc
import pandas as pd
from astropy.coordinates import SkyCoord


@dask.delayed
def cone_filter(data_frame: pd.DataFrame, ra, dec, radius, metadata: hc.catalog.Catalog):
    """Filters a dataframe to only include points within the specified cone

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        ra (float): Right Ascension of the center of the cone in degrees
        dec (float): Declination of the center of the cone in degrees
        radius (float): Radius of the cone in degrees
        metadata (hipscat.Catalog): hipscat `Catalog` with catalog_info that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the cone
    """
    df_ras = data_frame[metadata.catalog_info.ra_column].values
    df_decs = data_frame[metadata.catalog_info.dec_column].values
    df_coords = SkyCoord(df_ras, df_decs, unit='deg')
    center_coord = SkyCoord(ra, dec, unit='deg')
    df_separations = df_coords.separation(center_coord).value
    data_frame["_CONE_SEP"] = df_separations
    data_frame = data_frame.loc[data_frame["_CONE_SEP"] < radius]
    data_frame = data_frame.drop(columns=["_CONE_SEP"])
    return data_frame
