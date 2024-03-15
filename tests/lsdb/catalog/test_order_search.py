import pytest


def test_order_search(small_sky_source_catalog, assert_divisions_are_correct):
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


def test_order_search_eval_args(small_sky_source_catalog):
    with pytest.raises(ValueError, match="The minimum order should be less than or equal to maximum order"):
        small_sky_source_catalog.order_search(min_order=2, max_order=1)
