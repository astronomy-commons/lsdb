import nested_pandas as npd
from astropy.coordinates import SkyCoord
from hats.catalog import TableProperties
from hats.pixel_math.validators import validate_declination_values, validate_radius
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import HCCatalogTypeVar


class ConeSearch(AbstractSearch):
    """Perform a cone search to filter the catalog

    Filters to points within radius great circle distance to the point specified by ra and dec in degrees.
    Filters partitions in the catalog to those that have some overlap with the cone.
    """

    def __init__(self, ra: float, dec: float, radius_arcsec: float, fine: bool = True):
        super().__init__(fine)
        validate_radius(radius_arcsec)
        validate_declination_values(dec)
        self.ra = ra
        self.dec = dec
        self.radius_arcsec = radius_arcsec

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> MOC:
        """Filters catalog pixels according to the cone"""
        return hc_structure.filter_by_cone(self.ra, self.dec, self.radius_arcsec)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return cone_filter(frame, self.ra, self.dec, self.radius_arcsec, metadata)


def cone_filter(data_frame: npd.NestedFrame, ra, dec, radius_arcsec, metadata: TableProperties):
    """Filters a dataframe to only include points within the specified cone

    Args:
        data_frame (npd.NestedFrame): DataFrame containing points in the sky
        ra (float): Right Ascension of the center of the cone in degrees
        dec (float): Declination of the center of the cone in degrees
        radius_arcsec (float): Radius of the cone in arcseconds
        metadata (hc.TableProperties): hats `TableProperties` with metadata that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the cone
    """
    df_ras = data_frame[metadata.ra_column].to_numpy()
    df_decs = data_frame[metadata.dec_column].to_numpy()
    df_coords = SkyCoord(df_ras, df_decs, unit="deg")
    center_coord = SkyCoord(ra, dec, unit="deg")
    df_separations_deg = df_coords.separation(center_coord).value
    radius_degrees = radius_arcsec / 3600
    data_frame = data_frame.iloc[df_separations_deg < radius_degrees]
    return data_frame
