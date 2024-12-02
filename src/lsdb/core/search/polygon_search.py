import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
from hats.catalog import TableProperties
from hats.pixel_math.validators import validate_polygon
from lsst.sphgeom import ConvexPolygon, UnitVector3d

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import HCCatalogTypeVar


class PolygonSearch(AbstractSearch):
    """Perform a polygonal search to filter the catalog.

    Filters to points within the polygonal region specified in ra and dec, in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, vertices: list[tuple[float, float]], fine: bool = True):
        super().__init__(fine)
        validate_polygon(vertices)
        self.vertices = vertices
        self.polygon = get_cartesian_polygon(vertices)

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        """Filters catalog pixels according to the polygon"""
        return hc_structure.filter_by_polygon(self.vertices)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return polygon_filter(frame, self.polygon, metadata)


def polygon_filter(
    data_frame: npd.NestedFrame, polygon: ConvexPolygon, metadata: TableProperties
) -> npd.NestedFrame:
    """Filters a dataframe to only include points within the specified polygon.

    Args:
        data_frame (npd.NestedFrame): DataFrame containing points in the sky
        polygon (ConvexPolygon): Convex spherical polygon of interest, used to filter points
        metadata (hc.catalog.Catalog): hats `Catalog` with catalog_info that matches `dataframe`

    Returns:
        A new DataFrame with the rows from `dataframe` filtered to only the pixels inside the polygon.
    """
    ra_values = np.radians(data_frame[metadata.ra_column].to_numpy())
    dec_values = np.radians(data_frame[metadata.dec_column].to_numpy())
    inside_polygon = polygon.contains(ra_values, dec_values)
    data_frame = data_frame.iloc[inside_polygon]
    return data_frame


def get_cartesian_polygon(vertices: list[tuple[float, float]]) -> ConvexPolygon:
    """Creates the convex polygon to filter pixels with. It transforms the
    vertices, provided in sky coordinates of ra and dec, to their respective
    cartesian representation on the unit sphere.

    Args:
        vertices (list[tuple[float, float]): The list of vertices of the polygon
            to filter pixels with, as a list of (ra,dec) coordinates, in degrees.

    Returns:
        The convex polygon object.
    """
    vertices_xyz = hp.ang2vec(*np.array(vertices).T)
    edge_vectors = [UnitVector3d(x, y, z) for x, y, z in vertices_xyz]
    return ConvexPolygon(edge_vectors)
