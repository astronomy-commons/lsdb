import pandas as pd

import lsdb


def test_read_catalog_from_dataframe(small_sky_order1_catalog, small_sky_order1_csv):
    catalog = lsdb.read_dataframe(path=small_sky_order1_csv, catalog_name="my_catalog")
    assert isinstance(catalog, lsdb.Catalog)
    pd.testing.assert_frame_equal(
        catalog.hc_structure.get_pixels(), small_sky_order1_catalog.hc_structure.get_pixels()
    )
