import nested_dask as nd
import nested_pandas as npd
import numpy as np
import numpy.testing as npt
import pytest
from hats.pixel_math.validators import ValidatorsErrors

from lsdb.core.search.polygon_search import get_cartesian_polygon


def test_polygon_search_filters_correct_points(small_sky_order1_catalog, assert_divisions_are_correct):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon = get_cartesian_polygon(vertices)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    assert isinstance(polygon_search_catalog._ddf, nd.NestedFrame)
    polygon_search_df = polygon_search_catalog.compute()
    assert isinstance(polygon_search_df, npd.NestedFrame)
    ra_values_radians = np.radians(
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    )
    dec_values_radians = np.radians(
        polygon_search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    )
    assert all(polygon.contains(ra_values_radians, dec_values_radians))
    assert_divisions_are_correct(polygon_search_catalog)


def test_polygon_search_filters_correct_points_margin(
    small_sky_order1_source_with_margin, assert_divisions_are_correct
):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon = get_cartesian_polygon(vertices)
    polygon_search_catalog = small_sky_order1_source_with_margin.polygon_search(vertices)
    polygon_search_df = polygon_search_catalog.compute()
    ra_values_radians = np.radians(
        polygon_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.ra_column]
    )
    dec_values_radians = np.radians(
        polygon_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.dec_column]
    )
    assert all(polygon.contains(ra_values_radians, dec_values_radians))
    assert_divisions_are_correct(polygon_search_catalog)

    assert polygon_search_catalog.margin is not None
    polygon_search_margin_df = polygon_search_catalog.margin.compute()
    ra_values_radians = np.radians(
        polygon_search_margin_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.ra_column]
    )
    dec_values_radians = np.radians(
        polygon_search_margin_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.dec_column]
    )
    assert all(polygon.contains(ra_values_radians, dec_values_radians))
    assert_divisions_are_correct(polygon_search_catalog.margin)


def test_polygon_search_filters_partitions(small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    hc_polygon_search = small_sky_order1_catalog.hc_structure.filter_by_polygon(vertices)
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices, fine=False)
    assert len(hc_polygon_search.get_healpix_pixels()) == len(polygon_search_catalog.get_healpix_pixels())
    assert len(hc_polygon_search.get_healpix_pixels()) == polygon_search_catalog._ddf.npartitions
    for pixel in hc_polygon_search.get_healpix_pixels():
        assert pixel in polygon_search_catalog._ddf_pixel_map


def test_polygon_search_coarse_versus_fine(small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    coarse_polygon_search = small_sky_order1_catalog.polygon_search(vertices, fine=False)
    fine_polygon_search = small_sky_order1_catalog.polygon_search(vertices)
    assert coarse_polygon_search.get_healpix_pixels() == fine_polygon_search.get_healpix_pixels()
    assert coarse_polygon_search._ddf.npartitions == fine_polygon_search._ddf.npartitions
    assert len(coarse_polygon_search.compute()) > len(fine_polygon_search.compute())


def test_polygon_search_invalid_dec(small_sky_order1_catalog):
    # Some declination values are out of the [-90,90[ bounds
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_DEC):
        vertices = [(-20, 100), (-20, -1), (20, -1), (20, 100)]
        small_sky_order1_catalog.polygon_search(vertices)


def test_polygon_search_invalid_polygon(small_sky_order1_catalog):
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_NUM_VERTICES):
        vertices = [(100.1, -20.3), (100.1, 40.3)]
        small_sky_order1_catalog.polygon_search(vertices[:2])
    # The vertices should not have duplicates
    with pytest.raises(ValueError, match=ValidatorsErrors.DUPLICATE_VERTICES):
        vertices = [(100.1, -20.3), (100.1, -20.3), (280.1, -20.3), (280.1, 40.3)]
        small_sky_order1_catalog.polygon_search(vertices)
    # The polygons should not be on a great circle
    with pytest.raises(ValueError, match=ValidatorsErrors.DEGENERATE_POLYGON):
        vertices = [(100.1, 40.3), (100.1, -20.3), (280.1, -20.3), (280.1, 40.3)]
        small_sky_order1_catalog.polygon_search(vertices)
    with pytest.raises(ValueError, match=ValidatorsErrors.DEGENERATE_POLYGON):
        vertices = [(50.1, 0), (100.1, 0), (150.1, 0), (200.1, 0)]
        small_sky_order1_catalog.polygon_search(vertices)
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_CONCAVE_SHAPE):
        vertices = [(45, 30), (60, 60), (90, 45), (60, 50)]
        small_sky_order1_catalog.polygon_search(vertices)


def test_polygon_search_wrapped_right_ascension():
    """Tests the scenario where the polygon edges intersect the
    discontinuity of the RA [0,360] degrees range. For the same
    polygon we have several possible combination of coordinates
    (here with some float-fudging)."""
    vertices = [(-20.1, 1), (-20.2, -1), (20.3, -1)]
    all_vertices_combinations = [
        [(-20.1, 1), (-20.2, -1), (20.3, -1)],
        [(-20.1, 1), (339.8, -1), (20.3, -1)],
        [(-20.1, 1), (-380.2, -1), (20.3, -1)],
        [(-20.1, 1), (-20.2, -1), (380.3, -1)],
        [(-20.1, 1), (-20.2, -1), (-339.7, -1)],
        [(339.9, 1), (-20.2, -1), (20.3, -1)],
        [(339.9, 1), (339.8, -1), (20.3, -1)],
        [(339.9, 1), (-380.2, -1), (20.3, -1)],
        [(339.9, 1), (-20.2, -1), (380.3, -1)],
        [(339.9, 1), (-20.2, -1), (-339.7, -1)],
        [(-380.1, 1), (-20.2, -1), (20.3, -1)],
        [(-380.1, 1), (339.8, -1), (20.3, -1)],
        [(-380.1, 1), (-380.2, -1), (20.3, -1)],
        [(-380.1, 1), (-20.2, -1), (380.3, -1)],
        [(-380.1, 1), (-20.2, -1), (-339.7, -1)],
        [(-20.1, 1), (339.8, -1), (380.3, -1)],
        [(-20.1, 1), (339.8, -1), (-339.7, -1)],
        [(-20.1, 1), (-380.2, -1), (380.3, -1)],
        [(-20.1, 1), (-380.2, -1), (-339.7, -1)],
    ]
    polygon = get_cartesian_polygon(vertices)
    for v in all_vertices_combinations:
        polygon_2 = get_cartesian_polygon(v)
        npt.assert_allclose(polygon.getVertices(), polygon_2.getVertices(), rtol=1e-7)


def test_empty_polygon_search_with_margin(small_sky_order1_source_with_margin):
    vertices = [(80, 0), (100, 30), (120, 0)]
    polygon = small_sky_order1_source_with_margin.polygon_search(vertices)
    assert len(polygon._ddf_pixel_map) == 0
    assert len(polygon.margin._ddf_pixel_map) == 0
