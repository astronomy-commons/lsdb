import numpy as np


def test_box_search_ra(small_sky_order1_catalog, assert_divisions_are_correct):
    ra_search_catalog = small_sky_order1_catalog.box(ra=(280, 300))
    ra_search_df = ra_search_catalog.compute()
    ra_values = ra_search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    assert all(280 <= ra <= 300 for ra in ra_values)
    assert_divisions_are_correct(ra_search_catalog)


def test_box_search_ra_wrapped_values(small_sky_order1_catalog, assert_divisions_are_correct):
    ra_column = small_sky_order1_catalog.hc_structure.catalog_info.ra_column
    ra_search_catalog = small_sky_order1_catalog.box(ra=(-30, 30))
    ra_values = ra_search_catalog.compute()[ra_column]
    for ra_range in [(330, 30), (330, 390)]:
        ra_search_catalog = small_sky_order1_catalog.box(ra=ra_range)
        ra_search_df = ra_search_catalog.compute()
        assert all((0 <= ra <= 30 or 330 <= ra <= 360) for ra in ra_values)
        assert np.array_equal(ra_search_df[ra_column], ra_values)
        assert_divisions_are_correct(ra_search_catalog)


def test_box_search_dec(small_sky_order1_catalog, assert_divisions_are_correct):
    dec_search_catalog = small_sky_order1_catalog.box(dec=(0, 30))
    dec_search_df = dec_search_catalog.compute()
    dec_values = dec_search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    assert all(0 <= dec <= 30 for dec in dec_values)
    assert_divisions_are_correct(dec_search_catalog)


def test_box_search_polygon(small_sky_order1_catalog, assert_divisions_are_correct):
    search_catalog = small_sky_order1_catalog.box(ra=(280, 300), dec=(-40, -30))
    search_df = search_catalog.compute()
    ra_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    dec_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    assert all(280 <= ra <= 300 for ra in ra_values)
    assert all(-40 <= dec <= -30 for dec in dec_values)
    assert_divisions_are_correct(search_catalog)
