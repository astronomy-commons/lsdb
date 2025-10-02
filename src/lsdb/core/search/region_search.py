from __future__ import annotations

import astropy.units as u
import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy.visualization.wcsaxes import SphericalCircle, WCSAxes
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel, get_healpix_pixel, spatial_index
from hats.pixel_math.region_to_moc import wrap_ra_angles
from hats.pixel_math.validators import (
    validate_box,
    validate_declination_values,
    validate_polygon,
    validate_radius,
)
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import HCCatalogTypeVar


class BoxSearch(AbstractSearch):
    """Perform a box search to filter the catalog. This type of search is used for a
    range of right ascension or declination, where the right ascension edges follow
    great arc circles and the declination edges follow small arc circles.

    Filters to points within the ra / dec region, specified in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, ra: tuple[float, float], dec: tuple[float, float], fine: bool = True):
        super().__init__(fine)
        ra = tuple(wrap_ra_angles(ra)) if ra else None
        validate_box(ra, dec)
        self.ra, self.dec = ra, dec

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> MOC:
        """Filters catalog pixels according to the box"""
        return hc_structure.filter_by_box(self.ra, self.dec)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return box_filter(frame, self.ra, self.dec, metadata)


def box_filter(
    data_frame: npd.NestedFrame,
    ra: tuple[float, float],
    dec: tuple[float, float],
    metadata: TableProperties,
) -> npd.NestedFrame:
    """Filters a dataframe to only include points within the specified box region.

    Args:
        data_frame (npd.NestedFrame): DataFrame containing points in the sky
        ra (tuple[float, float]): Right ascension range, in degrees
        dec (tuple[float, float]): Declination range, in degrees
        metadata (hc.catalog.Catalog): hats `Catalog` with catalog_info that matches `data_frame`

    Returns:
        A new DataFrame with the rows from `data_frame` filtered to only the points inside the box region.
    """
    ra_values = data_frame[metadata.ra_column].to_numpy()
    dec_values = data_frame[metadata.dec_column].to_numpy()
    wrapped_ra = wrap_ra_angles(ra_values)
    mask_ra = _create_ra_mask(ra, wrapped_ra)
    mask_dec = (dec[0] <= dec_values) & (dec_values <= dec[1])
    data_frame = data_frame.iloc[mask_ra & mask_dec]
    return data_frame


def _create_ra_mask(ra: tuple[float, float], values: np.ndarray) -> np.ndarray:
    """Creates the mask to filter right ascension values. If this range crosses
    the discontinuity line (0 degrees), we have a branched logical operation."""
    if ra[0] == ra[1]:
        return np.ones(len(values), dtype=bool)
    if ra[0] < ra[1]:
        mask = (values >= ra[0]) & (values <= ra[1])
    else:
        mask = ((values >= ra[0]) & (values <= 360)) | ((values >= 0) & (values <= ra[1]))
    return mask


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

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> MOC:
        """Filters catalog pixels according to the cone"""
        return hc_structure.filter_by_cone(self.ra, self.dec, self.radius_arcsec)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return cone_filter(frame, self.ra, self.dec, self.radius_arcsec, metadata)

    def _perform_plot(self, ax: WCSAxes, **kwargs):
        kwargs_to_use = {"ec": "tab:red", "fc": "none"}
        kwargs_to_use.update(kwargs)

        circle = SphericalCircle(
            (self.ra * u.deg, self.dec * u.deg),
            self.radius_arcsec * u.arcsec,
            transform=ax.get_transform("icrs"),
            **kwargs_to_use,
        )
        ax.add_patch(circle)


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
    ra_rad = np.radians(data_frame[metadata.ra_column].to_numpy())
    dec_rad = np.radians(data_frame[metadata.dec_column].to_numpy())
    ra0 = np.radians(ra)
    dec0 = np.radians(dec)

    cos_angle = np.sin(dec_rad) * np.sin(dec0) + np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)

    # Clamp to valid range to avoid numerical issues
    cos_separation = np.clip(cos_angle, -1.0, 1.0)

    cos_radius = np.cos(np.radians(radius_arcsec / 3600))
    data_frame = data_frame[cos_separation >= cos_radius]
    return data_frame


class MOCSearch(AbstractSearch):
    """Filter the catalog by a MOC.

    Filters partitions in the catalog to those that are in a specified moc.
    """

    def __init__(self, moc: MOC, fine: bool = True):
        super().__init__(fine)
        self.moc = moc

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_by_moc(self.moc)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        df_ras = frame[metadata.ra_column].to_numpy()
        df_decs = frame[metadata.dec_column].to_numpy()
        mask = self.moc.contains_lonlat(df_ras * u.deg, df_decs * u.deg)
        return frame.iloc[mask]


class OrderSearch(AbstractSearch):
    """Filter the catalog by HEALPix order.

    Filters partitions in the catalog to those that are in the orders specified.
    Does not filter points inside those partitions.
    """

    def __init__(self, min_order: int = 0, max_order: int | None = None):
        super().__init__(fine=False)
        if max_order and min_order > max_order:
            raise ValueError("The minimum order should be lower than or equal to the maximum order")
        self.min_order = min_order
        self.max_order = max_order

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        max_catalog_order = hc_structure.pixel_tree.get_max_depth()
        max_order = max_catalog_order if self.max_order is None else self.max_order
        if self.min_order > max_order:
            raise ValueError("The minimum order is higher than the catalog's maximum order")
        pixels = [p for p in hc_structure.get_healpix_pixels() if self.min_order <= p.order <= max_order]
        return hc_structure.filter_from_pixel_list(pixels)

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        """Determine the search results within a data frame."""
        return frame


class PixelSearch(AbstractSearch):
    """Filter the catalog by HEALPix pixels.

    Filters partitions in the catalog to those that are in a specified pixel set.
    Does not filter points inside those partitions.
    """

    def __init__(self, pixels: tuple[int, int] | HealpixPixel | list[tuple[int, int] | HealpixPixel]):
        super().__init__(fine=False)
        if isinstance(pixels, tuple):
            self.pixels = [get_healpix_pixel(pixels)]
        elif isinstance(pixels, HealpixPixel):
            self.pixels = [pixels]
        elif pd.api.types.is_list_like(pixels):
            if len(pixels) == 0:
                raise ValueError("Some pixels required for PixelSearch")
            self.pixels = [get_healpix_pixel(pix) for pix in pixels]
        else:
            raise ValueError("Unsupported input for PixelSearch")

    @classmethod
    def from_radec(cls, ra: float | list[float], dec: float | list[float]) -> PixelSearch:
        """Create a pixel search region, based on radec points.

        Args:
            ra (float|list[float]): celestial coordinates, right ascension in degrees
            dec (float|list[float]): celestial coordinates, declination in degrees
        """
        pixels = list(spatial_index.compute_spatial_index(ra, dec))
        pixels = [(spatial_index.SPATIAL_INDEX_ORDER, pix) for pix in pixels]
        return cls(pixels)

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_from_pixel_list(self.pixels)

    def search_points(self, frame: npd.NestedFrame, _) -> npd.NestedFrame:
        return frame


class PolygonSearch(AbstractSearch):
    """Perform a polygonal search to filter the catalog.

    IMPORTANT: Requires additional ``lsst-sphgeom`` package

    Filters to points within the polygonal region specified in ra and dec, in degrees.
    Filters partitions in the catalog to those that have some overlap with the region.
    """

    def __init__(self, vertices: list[tuple[float, float]], fine: bool = True):
        super().__init__(fine)
        validate_polygon(vertices)
        self.vertices = vertices
        self.polygon = get_cartesian_polygon(vertices)

    def perform_hc_catalog_filter(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        """Filters catalog pixels according to the polygon"""
        return hc_structure.filter_by_polygon(self.vertices)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        """Determine the search results within a data frame"""
        return polygon_filter(frame, self.polygon, metadata)


def polygon_filter(data_frame: npd.NestedFrame, polygon, metadata: TableProperties) -> npd.NestedFrame:
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


# pylint: disable=import-outside-toplevel
def get_cartesian_polygon(vertices: list[tuple[float, float]]):
    """Creates the convex polygon to filter pixels with. It transforms the
    vertices, provided in sky coordinates of ra and dec, to their respective
    cartesian representation on the unit sphere.

    Args:
        vertices (list[tuple[float, float]): The list of vertices of the polygon
            to filter pixels with, as a list of (ra,dec) coordinates, in degrees.

    Returns:
        The convex polygon object.
    """
    from lsst.sphgeom import ConvexPolygon, UnitVector3d  # pylint: disable=import-error

    vertices_xyz = hp.ang2vec(*np.array(vertices).T)
    edge_vectors = [UnitVector3d(x, y, z) for x, y, z in vertices_xyz]
    return ConvexPolygon(edge_vectors)
