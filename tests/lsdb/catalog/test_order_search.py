import pandas as pd
import pytest

from lsdb.core.search import OrderSearch


def test_order_search_filters_correct_pixels(small_sky_source_catalog, assert_divisions_are_correct):
    order_search_catalog = small_sky_source_catalog.order_search(min_order=1, max_order=1)
    pixel_orders = [pixel.order for pixel in order_search_catalog.get_healpix_pixels()]
    assert all(order == 1 for order in pixel_orders)
    assert_divisions_are_correct(order_search_catalog)

    order_search_catalog = small_sky_source_catalog.order_search(min_order=1, max_order=2)
    pixel_orders = [pixel.order for pixel in order_search_catalog.get_healpix_pixels()]
    assert all(1 <= order <= 2 for order in pixel_orders)
    assert_divisions_are_correct(order_search_catalog)

    order_search_catalog = small_sky_source_catalog.order_search(min_order=1)
    pixel_orders = [pixel.order for pixel in order_search_catalog.get_healpix_pixels()]
    assert all(1 <= order <= 2 for order in pixel_orders)
    assert_divisions_are_correct(order_search_catalog)

    order_search_catalog = small_sky_source_catalog.order_search(max_order=1)
    pixel_orders = [pixel.order for pixel in order_search_catalog.get_healpix_pixels()]
    assert all(0 <= order <= 1 for order in pixel_orders)
    assert_divisions_are_correct(order_search_catalog)


def test_order_search_keeps_all_points(small_sky_source_catalog):
    metadata = small_sky_source_catalog.hc_structure
    partition_df = small_sky_source_catalog._ddf.partitions[0].compute()
    search = OrderSearch(min_order=1, max_order=2)
    filtered_df = search.search_points(partition_df, metadata)
    pd.testing.assert_frame_equal(partition_df, filtered_df)


def test_order_search_invalid_args(small_sky_source_catalog):
    with pytest.raises(ValueError, match="lower than or equal to the maximum order"):
        small_sky_source_catalog.order_search(min_order=2, max_order=1)
    with pytest.raises(ValueError, match="minimum order is higher than"):
        small_sky_source_catalog.order_search(min_order=3)
