import nested_dask as nd
import numpy as np
import pandas as pd

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


def test_dropna_on_nested(small_sky_with_nested_sources):
    def add_na_values_nested(df):
        """replaces the first source_ra value in each nested df with NaN"""
        for i in range(len(df)):
            first_ra_value = df.iloc[i]["sources"].iloc[0]["source_ra"]
            df["sources"].array[i] = df["sources"].array[i].replace(first_ra_value, np.NaN)
        return df

    filtered_cat = small_sky_with_nested_sources.map_partitions(add_na_values_nested)
    drop_na_cat = filtered_cat.dropna(on_nested="sources")
    assert isinstance(drop_na_cat, Catalog)
    assert isinstance(drop_na_cat._ddf, nd.NestedFrame)
    drop_na_sources_compute = drop_na_cat["sources"].compute()
    filtered_sources_compute = filtered_cat["sources"].compute()
    assert len(drop_na_sources_compute) == len(filtered_sources_compute)
    assert sum(map(len, drop_na_sources_compute)) < sum(map(len, filtered_sources_compute))
    pd.testing.assert_frame_equal(
        drop_na_cat.compute(), filtered_cat._ddf.dropna(on_nested="sources").compute()
    )


def test_nest_lists(small_sky_with_nested_sources):
    """Test the behavior of catalog.nest_lists"""
    cat_ndf = small_sky_with_nested_sources._ddf
    catlists_ndf = cat_ndf.sources.nest.to_lists()
    smallsky_lists = cat_ndf[["id", "ra", "dec"]].join(catlists_ndf)
    small_sky_with_nested_sources._ddf = smallsky_lists
    cat_ndf_renested = small_sky_with_nested_sources.nest_lists(base_columns=["id", "ra", "dec"])
