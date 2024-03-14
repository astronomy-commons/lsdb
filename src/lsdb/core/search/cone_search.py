from typing import List

import dask
import hipscat as hc
import pandas as pd
from astropy.coordinates import SkyCoord
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.cone_filter import filter_pixels_by_cone
from hipscat.pixel_math.validators import validate_declination_values, validate_radius
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.core.search.abstract_search import AbstractSearch


class ConeSearch(AbstractSearch):
    """Perform a cone search to filter the catalog

    Filters to points within radius great circle distance to the point specified by ra and dec in degrees.
    Filters partitions in the catalog to those that have some overlap with the cone.
    """

    def __init__(self, ra, dec, radius_arcsec):
        validate_radius(radius_arcsec)
        validate_declination_values(dec)

        self.ra = ra
        self.dec = dec
        self.radius_arcsec = radius_arcsec

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""
        pixel_tree = PixelTreeBuilder.from_healpix(pixels)
        return filter_pixels_by_cone(pixel_tree, self.ra, self.dec, self.radius_arcsec)

    def search_points(self, frame: pd.DataFrame, metadata: hc.catalog.Catalog) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        return cone_filter(frame, self.ra, self.dec, self.radius_arcsec, metadata)


@dask.delayed
def cone_filter(data_frame: pd.DataFrame, ra, dec, radius_arcsec, metadata: hc.catalog.Catalog):
    """Filters a dataframe to only include points within the specified cone

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        ra (float): Right Ascension of the center of the cone in degrees
        dec (float): Declination of the center of the cone in degrees
        radius_arcsec (float): Radius of the cone in arcseconds
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the cone
    """
    df_ras = data_frame[metadata.catalog_info.ra_column].values
    df_decs = data_frame[metadata.catalog_info.dec_column].values
    df_coords = SkyCoord(df_ras, df_decs, unit="deg")
    center_coord = SkyCoord(ra, dec, unit="deg")
    df_separations_deg = df_coords.separation(center_coord).value
    radius_degrees = radius_arcsec / 3600
    data_frame = data_frame.iloc[df_separations_deg < radius_degrees]
    return data_frame
