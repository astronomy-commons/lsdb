import pandas as pd
import pytest
from hats.pixel_math import HealpixPixel

import lsdb.nested as nd
from lsdb.core.search.region_search import PixelSearch


def test_pixel_search(small_sky_catalog, small_sky_order1_catalog):
    # Searching for pixels at a higher order
    catalog = small_sky_catalog.pixel_search([(1, 44), (1, 45)])
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert 1 == len(catalog._ddf_pixel_map)
    assert [HealpixPixel(0, 11)] == catalog.get_healpix_pixels()
    # Searching for pixels at a lower order
    catalog = small_sky_order1_catalog.pixel_search([(0, 11)])
    assert 4 == len(catalog._ddf_pixel_map)
    assert [
        HealpixPixel(1, 44),
        HealpixPixel(1, 45),
        HealpixPixel(1, 46),
        HealpixPixel(1, 47),
    ] == catalog.get_healpix_pixels()


def test_pixel_search_from_radec(small_sky_catalog, small_sky_order1_catalog):
    """Searching for pixels, based on radec values"""
    ## First, try with just one radec value
    search_object = PixelSearch.from_radec(282.5, -58.5)
    assert len(search_object.pixels) == 1

    search_result = small_sky_catalog.search(search_object)
    assert [HealpixPixel(0, 11)] == search_result.get_healpix_pixels()

    search_result = small_sky_order1_catalog.search(search_object)
    assert [HealpixPixel(1, 46)] == search_result.get_healpix_pixels()

    ## Two radec values in the same catalog pixel
    search_object = PixelSearch.from_radec([282.5, 283.5], [-58.5, -61.5])
    assert len(search_object.pixels) == 2

    search_result = small_sky_catalog.search(search_object)
    assert [HealpixPixel(0, 11)] == search_result.get_healpix_pixels()

    search_result = small_sky_order1_catalog.search(search_object)
    assert [HealpixPixel(1, 46)] == search_result.get_healpix_pixels()

    ## radec values from every order 1 pixel in the catalog
    search_object = PixelSearch.from_radec(
        [282.5, 283.5, 308.5, 346.5, 319.5], [-58.5, -61.5, -69.5, -60.5, -35.5]
    )
    assert len(search_object.pixels) == 5

    search_result = small_sky_catalog.search(search_object)
    assert [HealpixPixel(0, 11)] == search_result.get_healpix_pixels()

    search_result = small_sky_order1_catalog.search(search_object)
    assert [
        HealpixPixel(1, 44),
        HealpixPixel(1, 45),
        HealpixPixel(1, 46),
        HealpixPixel(1, 47),
    ] == search_result.get_healpix_pixels()


def test_pixel_search_types():
    """Check that we can initialize PixelSearch with a variety of types."""
    search_object = PixelSearch([(1, 44)])
    assert search_object.pixels == [HealpixPixel(1, 44)]

    search_object = PixelSearch((1, 44))
    assert search_object.pixels == [HealpixPixel(1, 44)]

    search_object = PixelSearch([HealpixPixel(1, 44)])
    assert search_object.pixels == [HealpixPixel(1, 44)]

    search_object = PixelSearch(HealpixPixel(1, 44))
    assert search_object.pixels == [HealpixPixel(1, 44)]

    with pytest.raises(ValueError, match="pixels required"):
        PixelSearch([])

    with pytest.raises(ValueError, match="Unsupported input"):
        PixelSearch(PixelSearch((1, 44)))


def test_pixel_search_is_empty(small_sky_catalog, small_sky_order1_catalog):
    catalog = small_sky_catalog.pixel_search([(1, 50)])
    assert 0 == len(catalog._ddf_pixel_map)
    catalog = small_sky_order1_catalog.pixel_search([(0, 10)])
    assert 0 == len(catalog._ddf_pixel_map)
    with pytest.raises(ValueError, match="pixels required"):
        small_sky_catalog.pixel_search([])


def test_pixel_search_keeps_all_points(small_sky_order1_catalog):
    metadata = small_sky_order1_catalog.hc_structure
    partition_df = small_sky_order1_catalog.get_partition(1, 44).compute()
    filtered_df = PixelSearch([(1, 44)]).search_points(partition_df, metadata)
    pd.testing.assert_frame_equal(partition_df, filtered_df)
