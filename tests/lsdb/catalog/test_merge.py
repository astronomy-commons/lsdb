import dask.dataframe as dd
import pandas as pd
import pytest


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_catalog_merge_on_indices(small_sky_catalog, small_sky_order1_catalog, how):
    kwargs = {"how": how, "left_index": True, "right_index": True, "suffixes": ("_left", "_right")}
    # Setting the object "id" for index on both catalogs
    small_sky_catalog._ddf = small_sky_catalog._ddf.set_index("id")
    small_sky_order1_catalog._ddf = small_sky_order1_catalog._ddf.set_index("id")
    # The wrapper outputs the same result as the underlying pandas merge
    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, **kwargs)
    assert isinstance(merged_ddf, dd.core.DataFrame)
    expected_df = small_sky_catalog._ddf.merge(small_sky_order1_catalog._ddf, **kwargs)
    pd.testing.assert_frame_equal(expected_df.compute(), merged_ddf.compute())


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_catalog_merge_on_columns(small_sky_catalog, small_sky_order1_catalog, how):
    kwargs = {"how": how, "on": "id", "suffixes": ("_left", "_right")}
    # Make sure none of the test catalogs have "id" for index
    small_sky_catalog._ddf = small_sky_catalog._ddf.reset_index()
    small_sky_order1_catalog._ddf = small_sky_order1_catalog._ddf.reset_index()
    # The wrapper outputs the same result as the underlying pandas merge
    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, **kwargs)
    assert isinstance(merged_ddf, dd.core.DataFrame)
    expected_df = small_sky_catalog._ddf.merge(small_sky_order1_catalog._ddf, **kwargs)
    pd.testing.assert_frame_equal(expected_df.compute(), merged_ddf.compute())


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_catalog_merge_on_index_and_column(small_sky_catalog, small_sky_order1_catalog, how):
    kwargs = {"how": how, "left_index": True, "right_on": "id", "suffixes": ("_left", "_right")}
    # Setting the object "id" for index on the left catalog
    small_sky_catalog._ddf = small_sky_catalog._ddf.set_index("id")
    # Make sure the right catalog does not have "id" for index
    small_sky_order1_catalog._ddf = small_sky_order1_catalog._ddf.reset_index()
    # The wrapper outputs the same result as the underlying pandas merge
    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, **kwargs)
    assert isinstance(merged_ddf, dd.core.DataFrame)
    expected_df = small_sky_catalog._ddf.merge(small_sky_order1_catalog._ddf, **kwargs)
    pd.testing.assert_frame_equal(expected_df.compute(), merged_ddf.compute())


def test_catalog_merge_invalid_suffixes(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError, match="`suffixes` must be a tuple with two strings"):
        small_sky_catalog.merge(
            small_sky_order1_catalog, how="inner", on="id", suffixes=("_left", "_middle", "_right")
        )


def test_catalog_merge_no_suffixes(small_sky_catalog, small_sky_order1_catalog):
    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how="inner", on="id")
    assert isinstance(merged_ddf, dd.core.DataFrame)
    # Get the columns with the same name in both catalogs
    non_join_columns_left = small_sky_catalog._ddf.columns.drop("id")
    non_join_columns_right = small_sky_order1_catalog._ddf.columns.drop("id")
    intersected_cols = list(set(non_join_columns_left) & set(non_join_columns_right))
    # The suffixes of these columns in the dataframe include the catalog names
    suffixes = [f"_{small_sky_catalog.name}", f"_{small_sky_order1_catalog.name}"]
    for column in intersected_cols:
        for suffix in suffixes:
            assert f"{column}{suffix}" in merged_ddf.columns
