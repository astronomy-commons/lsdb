import nested_dask as nd
import pandas as pd
from hats.pixel_math import HealpixPixel

from lsdb.core.search.pixel_search import PixelSearch


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


def test_pixel_search_is_empty(small_sky_catalog, small_sky_order1_catalog):
    catalog = small_sky_catalog.pixel_search([(1, 50)])
    assert 0 == len(catalog._ddf_pixel_map)
    catalog = small_sky_order1_catalog.pixel_search([(0, 10)])
    assert 0 == len(catalog._ddf_pixel_map)
    catalog = small_sky_catalog.pixel_search([])
    assert 0 == len(catalog._ddf_pixel_map)


def test_pixel_search_keeps_all_points(small_sky_order1_catalog):
    metadata = small_sky_order1_catalog.hc_structure
    partition_df = small_sky_order1_catalog.get_partition(1, 44).compute()
    filtered_df = PixelSearch([(1, 44)]).search_points(partition_df, metadata)
    pd.testing.assert_frame_equal(partition_df, filtered_df)
