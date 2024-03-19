import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from hipscat.pixel_math.validators import ValidatorsErrors


def test_cone_search_filters_correct_points(small_sky_order1_catalog, assert_divisions_are_correct):
    ra = 0
    dec = -80
    radius_degrees = 20
    radius = radius_degrees * 3600
    center_coord = SkyCoord(ra, dec, unit="deg")
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius)
    cone_search_df = cone_search_catalog.compute()
    for _, row in small_sky_order1_catalog.compute().iterrows():
        row_ra = row[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
        row_dec = row[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
        sep = SkyCoord(row_ra, row_dec, unit="deg").separation(center_coord)
        if sep.degree <= radius_degrees:
            assert len(cone_search_df.loc[cone_search_df["id"] == row["id"]]) == 1
        else:
            assert len(cone_search_df.loc[cone_search_df["id"] == row["id"]]) == 0
    assert_divisions_are_correct(cone_search_catalog)


def test_cone_search_filters_correct_points_margin(
    small_sky_order1_source_with_margin,
    assert_divisions_are_correct,
    cone_search_expected,
    cone_search_margin_expected,
):
    ra = -35
    dec = -55
    radius_degrees = 2
    radius = radius_degrees * 3600
    cone_search_catalog = small_sky_order1_source_with_margin.cone_search(ra, dec, radius)
    assert cone_search_catalog.margin is not None
    cone_search_df = cone_search_catalog.compute()
    pd.testing.assert_frame_equal(cone_search_df, cone_search_expected, check_dtype=False)
    cone_search_margin_df = cone_search_catalog.margin.compute()
    pd.testing.assert_frame_equal(cone_search_margin_df, cone_search_margin_expected, check_dtype=False)
    assert_divisions_are_correct(cone_search_catalog)
    assert_divisions_are_correct(cone_search_catalog.margin)


def test_cone_search_big_margin(small_sky_order1_source_with_margin):
    small_sky_order1_source_with_margin.margin.hc_structure.catalog_info.margin_threshold = 600000
    with pytest.raises(ValueError, match="Margin size"):
        small_sky_order1_source_with_margin.cone_search(0, 0, 1)


def test_cone_search_filters_partitions(small_sky_order1_catalog):
    ra = 0
    dec = -80
    radius = 20 * 3600
    hc_conesearch = small_sky_order1_catalog.hc_structure.filter_by_cone(ra, dec, radius)
    consearch_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius, fine=False)
    assert len(hc_conesearch.get_healpix_pixels()) == len(consearch_catalog.get_healpix_pixels())
    assert len(hc_conesearch.get_healpix_pixels()) == consearch_catalog._ddf.npartitions
    for pixel in hc_conesearch.get_healpix_pixels():
        assert pixel in consearch_catalog._ddf_pixel_map


def test_cone_search_filters_no_matching_points(small_sky_order1_catalog, assert_divisions_are_correct):
    ra = 0
    dec = -80
    radius = 0.2 * 3600
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius)
    cone_search_df = cone_search_catalog.compute()
    assert len(cone_search_df) == 0
    assert_divisions_are_correct(cone_search_catalog)


def test_cone_search_filters_no_matching_partitions(small_sky_order1_catalog, assert_divisions_are_correct):
    ra = 20
    dec = 80
    radius = 20 * 3600
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra, dec, radius)
    cone_search_df = cone_search_catalog.compute()
    assert len(cone_search_df) == 0
    assert_divisions_are_correct(cone_search_catalog)


def test_cone_search_wrapped_ra(small_sky_order1_catalog):
    # RA is inside the [0,360] degree range
    small_sky_order1_catalog.cone_search(200.3, 0, 1.2)
    # RA is outside the [0,360] degree range, but they are wrapped
    small_sky_order1_catalog.cone_search(400.9, 0, 1.3)
    small_sky_order1_catalog.cone_search(-100.1, 0, 1.5)


def test_cone_search_coarse_versus_fine(small_sky_order1_catalog):
    ra = 0
    dec = -80
    radius = 20 * 3600  # 20 degrees
    coarse_cone_search = small_sky_order1_catalog.cone_search(ra, dec, radius, fine=False)
    fine_cone_search = small_sky_order1_catalog.cone_search(ra, dec, radius)
    assert coarse_cone_search.get_healpix_pixels() == fine_cone_search.get_healpix_pixels()
    assert coarse_cone_search._ddf.npartitions == fine_cone_search._ddf.npartitions
    assert len(coarse_cone_search.compute()) > len(fine_cone_search.compute())


def test_invalid_dec_and_negative_radius(small_sky_order1_catalog):
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_DEC):
        small_sky_order1_catalog.cone_search(0, -100.3, 1.2)
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_DEC):
        small_sky_order1_catalog.cone_search(0, 100.4, 1.3)
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADIUS):
        small_sky_order1_catalog.cone_search(0, 0, -1.5)
