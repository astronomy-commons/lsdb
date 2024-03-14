from typing import List, Tuple

import dask
import healpy as hp
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.polygon_filter import (
    CartesianCoordinates,
    SphericalCoordinates,
    filter_pixels_by_polygon,
)
from hipscat.pixel_math.validators import validate_declination_values, validate_polygon
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder
from lsst.sphgeom import ConvexPolygon, UnitVector3d

from lsdb.core.search.abstract_search import AbstractSearch


class PolygonSearch(AbstractSearch):
    """Perform a polygonal search to filter the catalog.

    Filters to points within the polygonal region specified in ra and dec, in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, vertices: List[SphericalCoordinates]):
        _, dec = np.array(vertices).T
        validate_declination_values(dec)
        self.polygon, self.vertices_xyz = get_cartesian_polygon(vertices)

    def search_partitions(self, pixels: List[HealpixPixel]) -> List[HealpixPixel]:
        """Determine the target partitions for further filtering."""
        pixel_tree = PixelTreeBuilder.from_healpix(pixels)
        return filter_pixels_by_polygon(pixel_tree, self.vertices_xyz)

    def search_points(self, frame: pd.DataFrame, metadata: hc.catalog.Catalog) -> pd.DataFrame:
        """Determine the search results within a data frame"""
        return polygon_filter(frame, self.polygon, metadata)


@dask.delayed
def polygon_filter(data_frame: pd.DataFrame, polygon: ConvexPolygon, metadata: hc.catalog.Catalog):
    """Filters a dataframe to only include points within the specified polygon.

    Args:
        data_frame (pd.DataFrame): DataFrame containing points in the sky
        polygon (ConvexPolygon): Convex spherical polygon of interest, used to filter points
        metadata (hc.catalog.Catalog): hipscat `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon.
    """
    ra_values = np.radians(data_frame[metadata.catalog_info.ra_column].values)
    dec_values = np.radians(data_frame[metadata.catalog_info.dec_column].values)
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
