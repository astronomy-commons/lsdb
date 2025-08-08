import os
import pytest
import lsdb
import pandas as pd
import nested_pandas as npd
import lsdb.nested as nd
from lsdb import ConeSearch

def test_concat_catalog_row_count(small_sky_order1_catalog, helpers):
    """
    Test that concatenating two catalogs results in a catalog whose row count
    is the sum of the individual catalogs' row counts, and that the structure is preserved.
    """
    # Define two cone regions with partial overlap
    cone1 = ConeSearch(325, -55, 36000)  # 10 deg radius
    cone2 = ConeSearch(325, -25, 36000)  # 10 deg radius

    # Perform cone searches
    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    # Concatenate the two catalogs
    concat_cat = left_cat.concat(right_cat)

    # Compute all results
    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    # Check that concat contains all rows from left and right
    expected_total = len(df_left) + len(df_right)
    actual_total = len(df_concat)

    assert actual_total == expected_total, (
        f"Expected {expected_total} rows after concat, but got {actual_total}"
    )

    # Check internal type and structure
    assert isinstance(concat_cat._ddf, nd.NestedFrame)
    assert isinstance(df_concat, npd.NestedFrame)

    # Optional: check if the structure metadata is still valid
    helpers.assert_divisions_are_correct(concat_cat)
    assert concat_cat.hc_structure.catalog_path is None

def test_concat_catalog_row_content(small_sky_order1_catalog):
    """
    Test that every row in the concatenated catalog matches exactly with the corresponding
    row (by 'id') in either the left or right catalog, and that all column values are identical.
    """
    # Define two cone regions with partial overlap
    cone1 = ConeSearch(325, -55, 36000)
    cone2 = ConeSearch(325, -25, 36000)

    # Perform cone searches
    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    # Concatenate the two catalogs
    concat_cat = left_cat.concat(right_cat)

    # Compute all results
    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    # For each row in df_concat, check if it exists in df_left or df_right and if all values match
    for idx, row in df_concat.iterrows():
        row_id = row["id"]
        # Find the corresponding row in df_left or df_right
        match_left = df_left[df_left["id"] == row_id]
        match_right = df_right[df_right["id"] == row_id]
        assert not (match_left.empty and match_right.empty), f"id {row_id} not found in left nor right"
        if not match_left.empty:
            expected_row = match_left.iloc[0]
        else:
            expected_row = match_right.iloc[0]
        # Compare all column values
        for col in df_concat.columns:
            assert row[col] == expected_row[col], f"Different value in column '{col}' for id {row_id}"

def test_concat_catalog_margin_content(small_sky_order1_collection_catalog):
    """
    Test that every row in the concatenated catalog's margin matches exactly with the corresponding
    row (by 'id') in either the left or right catalog's margin, and that all column values are identical.
    This test assumes the catalog structure and pixel order are the same for this specific case.
    """
    # Define two cone regions with partial overlap
    cone1 = ConeSearch(325, -55, 36000)
    cone2 = ConeSearch(325, -25, 36000)

    # Perform cone searches
    left_cat = small_sky_order1_collection_catalog.search(cone1)
    right_cat = small_sky_order1_collection_catalog.search(cone2)

    # Concatenate the two catalogs
    concat_cat = left_cat.concat(right_cat)

    # Compute margins
    margin_left = left_cat.margin.compute()
    margin_right = right_cat.margin.compute()
    margin_concat = concat_cat.margin.compute()

    # For each row in margin_concat, check if it exists in margin_left or margin_right and if all values match
    for idx, row in margin_concat.iterrows():
        row_id = row["id"]
        match_left = margin_left[margin_left["id"] == row_id]
        match_right = margin_right[margin_right["id"] == row_id]
        assert not (match_left.empty and match_right.empty), f"id {row_id} not found in left nor right margin"
        if not match_left.empty:
            expected_row = match_left.iloc[0]
        else:
            expected_row = match_right.iloc[0]
        for col in margin_concat.columns:
            assert row[col] == expected_row[col], f"Different value in column '{col}' for id {row_id} in margin"