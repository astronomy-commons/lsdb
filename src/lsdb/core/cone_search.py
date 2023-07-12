import dask
import hipscat as hc
import pandas as pd
from astropy.coordinates import SkyCoord


@dask.delayed
def cone_filter(df: pd.DataFrame, ra, dec, radius, metadata: hc.catalog.Catalog):
    df_ras = df[metadata.catalog_info.ra_column].values
    df_decs = df[metadata.catalog_info.dec_column].values
    df_coords = SkyCoord(df_ras, df_decs, unit='deg')
    center_coord = SkyCoord(ra, dec, unit='deg')
    df_separations = df_coords.separation(center_coord).value
    df["_CONE_SEP"] = df_separations
    df = df.loc[df["_CONE_SEP"] < radius]
    df = df.drop(columns=["_CONE_SEP"])
    return df
