import numpy as np
import pytest

from lsdb.core.search.polygon_search import get_cartesian_polygon


def test_polygon_search_filters_correct_points(small_sky_order1_catalog, assert_divisions_are_correct):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon, _ = get_cartesian_polygon(vertices)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    polygon_search_df = polygon_search_catalog.compute()
    ra_values_radians = np.radians(
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    )
    dec_values_radians = np.radians(
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    )
    assert all(polygon.contains(ra_values_radians, dec_values_radians))
    assert_divisions_are_correct(polygon_search_catalog)


def test_polygon_search_filters_partitions(small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    _, vertices_xyz = get_cartesian_polygon(vertices)
    hc_polygon_search = small_sky_order1_catalog.hc_structure.filter_by_polygon(vertices_xyz)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    assert len(hc_polygon_search.get_healpix_pixels()) == len(polygon_search_catalog.get_healpix_pixels())
    assert len(hc_polygon_search.get_healpix_pixels()) == polygon_search_catalog._ddf.npartitions
    for pixel in hc_polygon_search.get_healpix_pixels():
        assert pixel in polygon_search_catalog._ddf_pixel_map


def test_polygon_search_empty(small_sky_order1_catalog):
    vertices = [(0, 0), (1, 1), (0, 2)]
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    assert len(polygon_search_catalog.get_healpix_pixels()) == 0
    assert len(polygon_search_catalog.hc_structure.pixel_tree) == 1


def test_polygon_search_invalid_shape(small_sky_order1_catalog):
    """The polygon is not convex, so the shape is invalid"""
    vertices = [(0, 1), (1, 0), (1, 1), (0, 0), (1,1)]
    with pytest.raises(RuntimeError):
        small_sky_order1_catalog.polygon_search(vertices)
