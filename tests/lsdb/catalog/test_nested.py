import pandas as pd
import nested_dask as nd
from lsdb import Catalog


def test_dropna(small_sky_with_nested_sources):
    filtered_cat = small_sky_with_nested_sources.query("sources.mag < 15.1")
    drop_na_cat = filtered_cat.dropna()
    assert isinstance(drop_na_cat, Catalog)
    assert isinstance(drop_na_cat._ddf, nd.NestedFrame)
    drop_na_compute = drop_na_cat.compute()
    filtered_compute = filtered_cat.compute()
    assert len(drop_na_compute) < len(filtered_compute)
    pd.testing.assert_frame_equal(drop_na_compute, filtered_compute.dropna())
