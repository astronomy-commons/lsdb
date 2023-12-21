from typing import List, Tuple

import dask
import healpy as hp
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.pixel_math.polygon_filter import CartesianCoordinates, SphericalCoordinates
from lsst.sphgeom import ConvexPolygon, UnitVector3d


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
    data_frame["_INSIDE_POLYGON"] = polygon.contains(ra_values, dec_values)
    data_frame = data_frame.loc[data_frame["_INSIDE_POLYGON"]]
    data_frame = data_frame.drop(columns=["_INSIDE_POLYGON"])
    return data_frame


def get_cartesian_polygon(
    vertices: List[SphericalCoordinates],
) -> Tuple[ConvexPolygon, List[CartesianCoordinates]]:
    """Creates the convex polygon to filter pixels with. It transforms the vertices, provided
    in sky coordinates of ra and dec, to their respective cartesian representation on the unit sphere.

    Arguments:
        vertices (List[SphericalCoordinates]): The list of vertices of the polygon to
            filter pixels with, as a list of (ra,dec) coordinates, in degrees.

    Returns:
        A tuple, where the first element is the convex polygon object and the second is the
        list of cartesian coordinates of its vertices.
    """
    vertices_xyz = hp.ang2vec(*np.array(vertices).T, lonlat=True)
    vertices_vectors = [UnitVector3d(x, y, z) for x, y, z in vertices_xyz]
    return ConvexPolygon(vertices_vectors), vertices_xyz
