import hipscat as hc
import numpy.testing as npt

from lsdb.dask.divisions import get_pixels_divisions


def test_divisions_are_independent_of_pixel_order(small_sky_order1_catalog):
    hp_pixels = small_sky_order1_catalog.get_ordered_healpix_pixels()
    order = [p.order for p in hp_pixels]
    pixel = [p.pixel for p in hp_pixels]
    pixels = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(order, pixel)
    npt.assert_array_equal(pixels, sorted(pixels))
    # Calculate divisions for ordered pixels
    divisions = get_pixels_divisions(hp_pixels)
    # Swap the first two pixels and check that the computed divisions are the same
    hp_pixels[0], hp_pixels[1] = hp_pixels[1], hp_pixels[0]
    assert divisions == get_pixels_divisions(hp_pixels)
