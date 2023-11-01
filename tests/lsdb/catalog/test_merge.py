import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest


def test_catalog_merge_invalid_suffixes(small_sky_catalog, small_sky_order1_catalog):
    with pytest.raises(ValueError, match="`suffixes` must be a tuple with two strings"):
        small_sky_catalog.merge(
            small_sky_order1_catalog, how="inner", on="id", suffixes=("_left", "_middle", "_right")
        )


def test_catalog_merge_no_suffixes(small_sky_catalog, small_sky_order1_catalog):
    on = "id"
    how = "inner"

    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how=how, on=on)
    assert isinstance(merged_ddf, dd.DataFrame)

    # Columns in the merged dataframe have the catalog name as suffix
    non_join_columns_left = small_sky_catalog._ddf.columns.drop(on)
    non_join_columns_right = small_sky_order1_catalog._ddf.columns.drop(on)
    intersected_cols = list(set(non_join_columns_left) & set(non_join_columns_right))

    suffixes = [f"_{small_sky_catalog.name}", f"_{small_sky_order1_catalog.name}"]

    for column in intersected_cols:
        for suffix in suffixes:
            assert f"{column}{suffix}" in merged_ddf.columns


def test_catalog_inner_merge(small_sky_catalog, small_sky_order1_catalog):
    on = "id"
    how = "inner"
    suffixes = ("_left", "_right")

    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how=how, on=on, suffixes=suffixes)
    assert isinstance(merged_ddf, dd.DataFrame)

    merged_df = merged_ddf.compute()
    left_df = small_sky_catalog._ddf.compute()
    right_df = small_sky_order1_catalog._ddf.compute()

    # The join column matches the intersection of values on both dataframes
    on_intersected = pd.Series(list(set(left_df[on]) & set(right_df[on])))
    assert_series_match(merged_df[on], on_intersected)

    # The remaining columns come from the original dataframes
    non_join_columns_df = merged_df.drop(on, axis=1)
    assert_other_columns_in_parent_dataframes(non_join_columns_df, left_df, right_df, suffixes)


def test_catalog_outer_merge(small_sky_catalog, small_sky_order1_catalog):
    on = "id"
    how = "outer"
    suffixes = ("_left", "_right")

    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how=how, on=on, suffixes=suffixes)
    assert isinstance(merged_ddf, dd.DataFrame)

    merged_df = merged_ddf.compute()
    left_df = small_sky_catalog._ddf.compute()
    right_df = small_sky_order1_catalog._ddf.compute()

    # The join column matches the whole set of values on both dataframes
    on_joined = pd.concat([left_df[on], right_df[on]])
    assert_series_match(merged_df[on], on_joined)

    # The remaining columns come from the original dataframes
    non_join_columns_df = merged_df.drop(on, axis=1)
    assert_other_columns_in_parent_dataframes(non_join_columns_df, left_df, right_df, suffixes)


def test_catalog_left_merge(small_sky_catalog, small_sky_order1_catalog):
    on = "id"
    how = "left"
    suffixes = ("_left", "_right")

    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how=how, on=on, suffixes=suffixes)
    assert isinstance(merged_ddf, dd.DataFrame)

    merged_df = merged_ddf.compute()
    left_df = small_sky_catalog.compute()
    right_df = small_sky_order1_catalog._ddf.compute()

    # The join column matches the values on the left dataframe
    assert_series_match(merged_df[on], left_df[on])

    # The remaining columns come from the original dataframes
    non_join_columns_df = merged_df.drop(on, axis=1)
    assert_other_columns_in_parent_dataframes(non_join_columns_df, left_df, right_df, suffixes)


def test_catalog_right_merge(small_sky_catalog, small_sky_order1_catalog):
    on = "id"
    how = "right"
    suffixes = ("_left", "_right")

    merged_ddf = small_sky_catalog.merge(small_sky_order1_catalog, how=how, on=on, suffixes=suffixes)
    assert isinstance(merged_ddf, dd.DataFrame)

    merged_df = merged_ddf.compute()
    left_df = small_sky_catalog._ddf.compute()
    right_df = small_sky_order1_catalog._ddf.compute()

    # The join column matches the values on the right dataframe
    assert_series_match(merged_df[on], right_df[on])

    # The remaining columns come from the original dataframes
    non_join_columns_df = merged_df.drop(on, axis=1)
    assert_other_columns_in_parent_dataframes(non_join_columns_df, left_df, right_df, suffixes)


def assert_other_columns_in_parent_dataframes(non_join_columns_df, left_df, right_df, suffixes):
    """Ensures the columns of a merged dataframe have the expected provenience. If a column has
    a suffix, the original dataframes had the same named column. If the column name has no suffix,
    it is present in one of the dataframes, but not in both."""
    _left, _right = suffixes
    for col_name, _ in non_join_columns_df.items():
        if col_name.endswith(_left):
            original_col_name = col_name[: -len(_left)]
            assert_series_match(non_join_columns_df[col_name], left_df[original_col_name])
        elif col_name.endswith(_right):
            original_col_name = col_name[: -len(_right)]
            assert_series_match(non_join_columns_df[col_name], right_df[original_col_name])
        elif col_name in left_df.columns:
            assert col_name not in right_df.columns
            assert_series_match(non_join_columns_df[col_name], left_df[col_name])
        else:
            assert col_name in right_df.columns and col_name not in left_df.columns
            assert_series_match(non_join_columns_df[col_name], right_df[col_name])


def assert_series_match(series_1, series_2):
    """Checks if a pandas series matches another in value, ignoring duplicates."""
    sorted_unique_1 = np.sort(series_1.drop_duplicates().to_numpy(), axis=0)
    sorted_unique_2 = np.sort(series_2.drop_duplicates().to_numpy(), axis=0)
    assert np.array_equal(sorted_unique_1, sorted_unique_2)
