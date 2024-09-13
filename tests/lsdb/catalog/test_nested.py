import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd

from lsdb import Catalog


def test_dropna(small_sky_with_nested_sources):
    filtered_cat = small_sky_with_nested_sources.query("sources.mag < 15.1")
    drop_na_cat = filtered_cat.dropna()
    assert isinstance(drop_na_cat, Catalog)
    assert isinstance(drop_na_cat._ddf, nd.NestedFrame)
    drop_na_compute = drop_na_cat.compute()
    assert isinstance(drop_na_compute, npd.NestedFrame)
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


def test_reduce(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    pd.testing.assert_frame_equal(reduced_cat_compute, reduced_ddf.compute())
