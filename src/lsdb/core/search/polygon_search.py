import dask
import hipscat as hc
import pandas as pd
from spherical_geometry.polygon import SingleSphericalPolygon


@dask.delayed
def polygon_filter(data_frame: pd.DataFrame, polygon: SingleSphericalPolygon, metadata: hc.catalog.Catalog):
    """Filters a dataframe to only include points within the specified polygon.

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        polygon (SingleSphericalPolygon): spherical polygon of interest, used to filter points
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon.
    """
    data_frame["_INSIDE_POLYGON"] = [
        polygon.contains_radec(ra, dec)
        for ra, dec in zip(
            data_frame[metadata.catalog_info.ra_column].values,
            data_frame[metadata.catalog_info.dec_column].values,
        )
    ]
    data_frame = data_frame.loc[data_frame["_INSIDE_POLYGON"]]
    data_frame = data_frame.drop(columns=["_INSIDE_POLYGON"])
    return data_frame
