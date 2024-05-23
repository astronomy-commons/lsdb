import pandas as pd
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.pixel_search import PixelSearch


def test_pixel_search_keeps_all_points(small_sky_order1_catalog):
    metadata = small_sky_order1_catalog.hc_structure
    partition_df = small_sky_order1_catalog.get_partition(1, 44).compute()
    search = PixelSearch([HealpixPixel(1, 44)])
    filtered_df = search.search_points(partition_df, metadata)
    pd.testing.assert_frame_equal(partition_df, filtered_df)
