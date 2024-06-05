from typing import List, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math.polygon_filter import CartesianCoordinates, SphericalCoordinates, generate_polygon_moc
from hipscat.pixel_math.validators import validate_declination_values, validate_polygon
from lsst.sphgeom import ConvexPolygon, UnitVector3d
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch


class PolygonSearch(AbstractSearch):
    """Perform a polygonal search to filter the catalog.

    Filters to points within the polygonal region specified in ra and dec, in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, vertices: List[SphericalCoordinates], fine: bool = True):
        super().__init__(fine)
        _, dec = np.array(vertices).T
        validate_declination_values(dec)
        self.vertices = np.array(vertices)
        self.polygon, self.vertices_xyz = get_cartesian_polygon(vertices)

    def generate_search_moc(self, max_order: int) -> MOC:
        return generate_polygon_moc(self.vertices_xyz, max_order)

    def search_points(self, frame: pd.DataFrame, metadata: CatalogInfo) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        return polygon_filter(frame, self.polygon, metadata)


def polygon_filter(data_frame: pd.DataFrame, polygon: ConvexPolygon, metadata: CatalogInfo):
    """Filters a dataframe to only include points within the specified polygon.

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        polygon (ConvexPolygon): Convex spherical polygon of interest, used to filter points
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon.
    """
    ra_values = np.radians(data_frame[metadata.ra_column].to_numpy())
    dec_values = np.radians(data_frame[metadata.dec_column].to_numpy())
    inside_polygon = polygon.contains(ra_values, dec_values)
    data_frame = data_frame.iloc[inside_polygon]
    return data_frame


def get_cartesian_polygon(
    vertices: List[SphericalCoordinates],
) -> Tuple[ConvexPolygon, List[CartesianCoordinates]]:
    """Creates the convex polygon to filter pixels with. It transforms the vertices, provided
    in sky coordinates of ra and dec, to their respective cartesian representation on the unit sphere.

    Args:
        vertices (List[Tuple[float, float]): The list of vertices of the polygon to
            filter pixels with, as a list of (ra,dec) coordinates, in degrees.

    Returns:
        A tuple, where the first element is the convex polygon object and the second is the
        list of cartesian coordinates of its vertices.
    """
    vertices_xyz = hp.ang2vec(*np.array(vertices).T, lonlat=True)
    validate_polygon(vertices_xyz)
    edge_vectors = [UnitVector3d(x, y, z) for x, y, z in vertices_xyz]
    return ConvexPolygon(edge_vectors), vertices_xyz
