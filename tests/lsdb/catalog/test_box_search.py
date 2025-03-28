import numpy as np
import pytest
from hats.pixel_math.validators import ValidatorsErrors


def test_box_search_filters_correct_points(small_sky_order1_catalog, helpers):
    search_catalog = small_sky_order1_catalog.box_search(ra=(280, 300), dec=(-40, -30))
    search_df = search_catalog.compute()
    ra_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    dec_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    assert len(search_df) < len(small_sky_order1_catalog.compute())
    assert all(280 <= ra <= 300 for ra in ra_values)
    assert all(-40 <= dec <= -30 for dec in dec_values)
    helpers.assert_divisions_are_correct(search_catalog)


def test_box_search_filters_correct_points_margin(small_sky_order1_source_with_margin, helpers):
    ra_search_catalog = small_sky_order1_source_with_margin.box_search(ra=(280, 300), dec=(-90, 30))
    ra_search_df = ra_search_catalog.compute()
    ra_values = ra_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.ra_column]
    dec_values = ra_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.dec_column]
    assert len(ra_search_df) < len(small_sky_order1_source_with_margin.compute())
    assert all(280 <= ra <= 300 for ra in ra_values)
    assert all(-90 <= dec <= 30 for dec in dec_values)
    helpers.assert_divisions_are_correct(ra_search_catalog)

    assert ra_search_catalog.margin is not None
    ra_margin_search_df = ra_search_catalog.margin.compute()
    ra_values = ra_margin_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.ra_column]
    dec_values = ra_margin_search_df[small_sky_order1_source_with_margin.hc_structure.catalog_info.dec_column]
    assert len(ra_margin_search_df) < len(small_sky_order1_source_with_margin.margin.compute())
    assert all(280 <= ra <= 300 for ra in ra_values)
    assert all(-90 <= dec <= 30 for dec in dec_values)
    helpers.assert_divisions_are_correct(ra_search_catalog.margin)


def test_box_search_ra_complement(small_sky_order1_catalog):
    dec = (-90, 90)
    ra_column = small_sky_order1_catalog.hc_structure.catalog_info.ra_column

    ra_search_catalog = small_sky_order1_catalog.box_search(ra=(280, 300), dec=dec)
    filtered_ra_values = ra_search_catalog.compute()[ra_column]
    assert len(filtered_ra_values) == 34

    # The complement search contains the remaining catalog points
    complement_search_catalog = small_sky_order1_catalog.box_search(ra=(300, 280), dec=dec)
    complement_search_ra_values = complement_search_catalog.compute()[ra_column]
    assert len(complement_search_ra_values) == 97

    joined_values = np.concatenate([filtered_ra_values, complement_search_ra_values])
    all_catalog_values = small_sky_order1_catalog.compute()[ra_column].to_numpy()
    assert np.array_equal(np.sort(joined_values), np.sort(all_catalog_values))


def test_box_search_ra_wrapped_filters_correct_points(small_sky_order1_catalog):
    dec = (-90, 90)
    ra_column = small_sky_order1_catalog.hc_structure.catalog_info.ra_column
    ra_search_catalog = small_sky_order1_catalog.box_search(ra=(330, 30), dec=dec)
    filtered_ra_values = ra_search_catalog.compute()[ra_column]
    # Some other options with values that need to be wrapped
    for ra_range in [(-30, 30), (330, 390), (330, -330)]:
        catalog = small_sky_order1_catalog.box_search(ra=ra_range, dec=dec)
        ra_values = catalog.compute()[ra_column]
        assert all((0 <= ra <= 30 or 330 <= ra <= 360) for ra in ra_values)
        assert np.array_equal(ra_values, filtered_ra_values)


def test_box_search_ra_boundary(small_sky_order1_catalog):
    dec = (-90, 90)
    ra_column = small_sky_order1_catalog.hc_structure.catalog_info.ra_column
    dec_column = small_sky_order1_catalog.hc_structure.catalog_info.dec_column

    ra_search_catalog = small_sky_order1_catalog.box_search(ra=(0, 0), dec=dec)
    ra_search_df = ra_search_catalog.compute()
    ra_values = ra_search_df[ra_column]
    dec_values = ra_search_df[dec_column]

    assert len(ra_search_df) > 0
    assert all((0 <= ra <= 360) for ra in ra_values)
    assert all((-90 <= dec <= 90) for dec in dec_values)

    for ra_range in [(0, 360), (360, 0)]:
        catalog_df = small_sky_order1_catalog.box_search(ra=ra_range, dec=dec).compute()
        assert np.array_equal(catalog_df[ra_column], ra_values)
        assert np.array_equal(catalog_df[dec_column], dec_values)


def test_box_search_filters_partitions(small_sky_order1_catalog):
    ra = (280, 300)
    dec = (-40, -30)
    hc_box_search = small_sky_order1_catalog.hc_structure.filter_by_box(ra, dec)
    box_search_catalog = small_sky_order1_catalog.box_search(ra, dec, fine=False)
    assert len(hc_box_search.get_healpix_pixels()) == len(box_search_catalog.get_healpix_pixels())
    assert len(hc_box_search.get_healpix_pixels()) == box_search_catalog._ddf.npartitions
    for pixel in hc_box_search.get_healpix_pixels():
        assert pixel in box_search_catalog._ddf_pixel_map


def test_box_search_coarse_versus_fine(small_sky_order1_catalog):
    ra = (280, 300)
    dec = (-40, -30)
    coarse_box_search = small_sky_order1_catalog.box_search(ra, dec, fine=False)
    fine_box_search = small_sky_order1_catalog.box_search(ra, dec)
    assert coarse_box_search.get_healpix_pixels() == fine_box_search.get_healpix_pixels()
    assert coarse_box_search._ddf.npartitions == fine_box_search._ddf.npartitions
    assert len(coarse_box_search.compute()) > len(fine_box_search.compute())


def test_box_search_invalid_args(small_sky_order1_catalog):
    # Some declination values are out of the [-90,90] bounds
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_DEC):
        small_sky_order1_catalog.box_search(ra=(0, 30), dec=(-100, -70))
    # Declination values should be in ascending order
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADEC_RANGE):
        small_sky_order1_catalog.box_search(ra=(0, 30), dec=(0, -10))
    # One or more ranges are defined with more than 2 values
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADEC_RANGE):
        small_sky_order1_catalog.box_search(ra=(0, 30), dec=(-40, -30, 10))
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADEC_RANGE):
        small_sky_order1_catalog.box_search(ra=(0, 30, 40), dec=(-40, 10))
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADEC_RANGE):
        small_sky_order1_catalog.box_search(ra=(0, 50), dec=(50, 50))
    # No range values were provided
    with pytest.raises(ValueError, match=ValidatorsErrors.INVALID_RADEC_RANGE):
        small_sky_order1_catalog.box_search(ra=None, dec=None)


def test_empty_box_search_with_margin(small_sky_order1_source_with_margin):
    ra = (100, 120)
    dec = (0, 10)
    box = small_sky_order1_source_with_margin.box_search(ra, dec, fine=False)
    assert len(box._ddf_pixel_map) == 0
    assert len(box.margin._ddf_pixel_map) == 0
