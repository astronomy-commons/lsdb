import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest

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
            df["sources"].array[i] = df["sources"].array[i].replace(first_ra_value, np.nan)
        return df

    filtered_cat = small_sky_with_nested_sources.map_partitions(add_na_values_nested)
    drop_na_cat = filtered_cat.dropna(on_nested="sources")
    assert isinstance(drop_na_cat, Catalog)
    assert isinstance(drop_na_cat._ddf, nd.NestedFrame)
    drop_na_sources_compute = drop_na_cat["sources"].compute()
    filtered_sources_compute = filtered_cat["sources"].compute()
    assert len(drop_na_sources_compute) == len(filtered_sources_compute)
    assert sum(map(len, drop_na_sources_compute)) < sum(map(len, filtered_sources_compute))


def test_nest_lists(small_sky_with_nested_sources):
    """Test the behavior of catalog.nest_lists"""
    cat_ndf = small_sky_with_nested_sources._ddf.map_partitions(
        lambda df: df.set_index(df.index.to_numpy() + np.arange(len(df)))
    )
    catlists_ndf = cat_ndf.sources.nest.to_lists()
    smallsky_lists = cat_ndf[["id", "ra", "dec"]].join(catlists_ndf)
    small_sky_with_nested_sources._ddf = smallsky_lists
    cat_ndf_renested = small_sky_with_nested_sources.nest_lists(base_columns=["id", "ra", "dec"])

    # check column structure
    assert "nested" in cat_ndf_renested.columns
    assert "id" in cat_ndf_renested.columns
    assert "ra" in cat_ndf_renested.columns
    assert "dec" in cat_ndf_renested.columns
    assert cat_ndf_renested._ddf["nested"].nest.fields == cat_ndf["sources"].nest.fields

    # try a compute call
    renested_flat = cat_ndf_renested.compute()["nested"].nest.to_flat()
    original_flat = cat_ndf.compute()["sources"].nest.to_flat()

    pd.testing.assert_frame_equal(renested_flat, original_flat)


def test_reduce(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    assert reduced_cat.hc_structure.catalog_info.ra_column == ""
    assert reduced_cat.hc_structure.catalog_info.dec_column == ""

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(
        mean_mag, "ra", "dec", "sources.mag", meta={"ra": float, "dec": float, "mean_mag": float}
    )

    pd.testing.assert_frame_equal(reduced_cat_compute, reduced_ddf.compute())


def test_reduce_append_columns(small_sky_with_nested_sources):
    def mean_mag(mag):
        return {"mean_mag": np.mean(mag)}

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={"mean_mag": float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(mean_mag, "sources.mag", meta={"mean_mag": float})

    pd.testing.assert_series_equal(reduced_cat_compute["mean_mag"], reduced_ddf.compute()["mean_mag"])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )


def test_reduce_no_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return np.mean(mag)

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={0: float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    reduced_cat_compute = reduced_cat.compute()
    assert isinstance(reduced_cat_compute, npd.NestedFrame)

    reduced_ddf = small_sky_with_nested_sources._ddf.reduce(mean_mag, "sources.mag", meta={0: float})

    pd.testing.assert_series_equal(reduced_cat_compute[0], reduced_ddf.compute()[0])
    pd.testing.assert_frame_equal(
        reduced_cat_compute[small_sky_with_nested_sources.columns], small_sky_with_nested_sources.compute()
    )


def test_reduce_invalid_return_column(small_sky_with_nested_sources):
    def mean_mag(mag):
        return pd.DataFrame.from_dict({"mean_mag": [np.mean(mag)]})

    reduced_cat = small_sky_with_nested_sources.reduce(
        mean_mag, "sources.mag", meta={0: float}, append_columns=True
    )

    assert isinstance(reduced_cat, Catalog)
    assert isinstance(reduced_cat._ddf, nd.NestedFrame)

    with pytest.raises(ValueError):
        reduced_cat.compute()


def test_reduce_append_columns_raises_error(small_sky_with_nested_sources):
    def mean_mag(ra, dec, mag):
        return {"ra": ra, "dec": dec, "mean_mag": np.mean(mag)}

    with pytest.raises(ValueError):
        small_sky_with_nested_sources.reduce(
            mean_mag,
            "ra",
            "dec",
            "sources.mag",
            meta={"ra": float, "dec": float, "mean_mag": float},
            append_columns=True,
        ).compute()
