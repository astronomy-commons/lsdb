import os
import pytest
import lsdb
import nested_pandas as npd
import lsdb.nested as nd
from lsdb import ConeSearch

def test_concat_catalog_row_count(small_sky_order1_catalog, helpers):
    # === Define two cone regions with partial overlap ===
    cone1 = ConeSearch(325, -55, 36000)  # 10 deg radius
    cone2 = ConeSearch(325, -25, 36000)  # 10 deg radius

    # === Perform cone searches ===
    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    # === Concatenate the two catalogs ===
    concat_cat = left_cat.concat(right_cat)

    # === Compute all results ===
    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    # === Check that concat contains all rows from left and right ===
    expected_total = len(df_left) + len(df_right)
    actual_total = len(df_concat)

    assert actual_total == expected_total, (
        f"Expected {expected_total} rows after concat, but got {actual_total}"
    )

    # === Check internal type and structure ===
    assert isinstance(concat_cat._ddf, nd.NestedFrame)
    assert isinstance(df_concat, npd.NestedFrame)

    # === Optional: check if the structure metadata is still valid ===
    helpers.assert_divisions_are_correct(concat_cat)
    assert concat_cat.hc_structure.catalog_path is None