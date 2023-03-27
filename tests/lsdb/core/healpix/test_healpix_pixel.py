import pytest

from lsdb.core.healpix.healpix_pixel import MAXIMUM_ORDER, HealpixPixel


def test_pixels_equal():
    order = 3
    pixel = 42
    pix1 = HealpixPixel(order=order, pixel=pixel)
    pix2 = HealpixPixel(order=order, pixel=pixel)
    assert pix1 == pix2
    assert pix1.order == pix2.order
    assert pix1.pixel == pix2.pixel


def test_order_greater_than_max_order_fails():
    with pytest.raises(ValueError):
        HealpixPixel(order=MAXIMUM_ORDER + 1, pixel=0)


def test_equal_pixel_hash_equal():
    order = 3
    pixel = 42
    test_string = "testing"
    pix1 = HealpixPixel(order=order, pixel=pixel)
    test_dict = {}
    test_dict[pix1] = test_string
    pix2 = HealpixPixel(order=order, pixel=pixel)
    assert pix1 == pix2
    assert pix2 in test_dict
    assert test_dict[pix2] == test_string
