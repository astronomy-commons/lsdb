import warnings
from collections import Counter
from types import SimpleNamespace
from typing import cast

import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb
import lsdb.nested as nd
from lsdb import ConeSearch
from lsdb.catalog.catalog import Catalog
from lsdb.dask import concat_catalog_data, merge_catalog_functions

# pylint: disable=too-many-lines

# ------------------------------- helpers ---------------------------------- #


def _normalize_na(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize null values in a DataFrame.

    Converts any null sentinel (NaN/pd.NA/None) to np.nan and casts all
    columns to "object" dtype. This ensures that `assert_frame_equal`
    compares nulls consistently regardless of dtype or backing.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame with consistent null handling.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype("object")
        out[c] = out[c].where(pd.notna(out[c]), np.nan)
    return out


def _row_multiset(df: pd.DataFrame) -> Counter:
    """Convert a DataFrame into a multiset of row tuples.

    Rows are converted into tuples and stored in a Counter. Row order is ignored,
    but multiplicity is preserved. Columns must already be aligned.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Counter: Multiset of canonicalized row tuples.
    """
    cols = sorted(df.columns)

    def _canon(v):
        if pd.isna(v):
            return None
        # Treat numeric values uniformly as float to avoid mismatches like 1 vs 1.0
        if isinstance(v, (int, np.integer, float, np.floating, np.number, bool, np.bool_)):
            return float(v)
        return v

    rows = (tuple(_canon(x) for x in row) for row in df[cols].itertuples(index=False, name=None))
    return Counter(rows)


def _align_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames to the union of their columns.

    Columns are sorted for determinism, and missing columns are filled with NaN.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames with aligned columns.
    """
    all_cols = sorted(set(df1.columns) | set(df2.columns))
    return df1.reindex(columns=all_cols), df2.reindex(columns=all_cols)


def _assert_concat_symmetry(
    left_cat: Catalog,
    right_cat: Catalog,
    cols_subset: list[str] | None = None,
    **concat_kwargs,
):
    """Assert that concatenation is symmetric between left and right catalogs.

    This checks that `left.concat(right, **concat_kwargs)` and
    `right.concat(left, **concat_kwargs)` produce the same content for both the
    main tables and the margin tables, regardless of row or column order.
    The comparison includes the spatial-index column.

    Args:
        left_cat (Catalog): Left catalog object.
        right_cat (Catalog): Right catalog object.
        cols_subset (list[str] | None): If provided, restrict comparison to this
            subset of columns (plus spatial index).
        **concat_kwargs: Keyword arguments forwarded to `Catalog.concat`.

    Raises:
        AssertionError: If the concatenated catalogs differ in content.
    """
    lr = left_cat.concat(right_cat, **concat_kwargs)
    rl = right_cat.concat(left_cat, **concat_kwargs)

    # Main tables
    df_lr = lr.compute().reset_index(drop=True)
    df_rl = rl.compute().reset_index(drop=True)

    # Margin tables (may be None)
    lr_margin = getattr(lr, "margin", None)
    rl_margin = getattr(rl, "margin", None)
    df_lr_margin = lr_margin.compute().reset_index(drop=True) if lr_margin is not None else None
    df_rl_margin = rl_margin.compute().reset_index(drop=True) if rl_margin is not None else None

    if cols_subset is not None:
        keep_main = [
            c for c in [SPATIAL_INDEX_COLUMN] + cols_subset if c in df_lr.columns or c in df_rl.columns
        ]
        df_lr = df_lr[keep_main]
        df_rl = df_rl[keep_main]

        if df_lr_margin is not None and df_rl_margin is not None:
            keep_margin = [
                c
                for c in [SPATIAL_INDEX_COLUMN] + cols_subset
                if c in df_lr_margin.columns or c in df_rl_margin.columns
            ]
            df_lr_margin = df_lr_margin[keep_margin]
            df_rl_margin = df_rl_margin[keep_margin]

    # Compare main tables
    df_lr, df_rl = _align_columns(df_lr, df_rl)
    ms_lr = _row_multiset(df_lr)
    ms_rl = _row_multiset(df_rl)
    assert ms_lr == ms_rl, "Main tables differ between left.concat(right) and right.concat(left)."

    # Compare margin tables (if both exist)
    if df_lr_margin is not None and df_rl_margin is not None:
        df_lr_margin, df_rl_margin = _align_columns(df_lr_margin, df_rl_margin)
        ms_lr_margin = _row_multiset(df_lr_margin)
        ms_rl_margin = _row_multiset(df_rl_margin)
        assert (
            ms_lr_margin == ms_rl_margin
        ), "Margin tables differ between left.concat(right) and right.concat(left)."


# --------------------------------- tests --------------------------------- #


def test_concat_catalog_row_count(small_sky_order1_catalog, helpers):
    """Verify that row count after concatenation equals the sum of both catalogs.

    This test ensures that concatenating two catalogs produces a catalog whose
    row count equals the sum of the input catalogs, and that the structure and
    internal types are preserved.

    Args:
        small_sky_order1_catalog: A small test catalog fixture.
        helpers: Utility fixture with helper assertions.
    """
    cone1 = ConeSearch(325, -55, 36000)  # 10 deg radius
    cone2 = ConeSearch(325, -25, 36000)  # 10 deg radius

    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    concat_cat = left_cat.concat(right_cat)

    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    expected_total = len(df_left) + len(df_right)
    actual_total = len(df_concat)
    assert (
        actual_total == expected_total
    ), f"Expected {expected_total} rows after concat, but got {actual_total}"

    # Internal types
    assert isinstance(concat_cat._ddf, nd.NestedFrame)
    assert isinstance(df_concat, npd.NestedFrame)

    # Structure/divisions sanity
    helpers.assert_divisions_are_correct(concat_cat)
    assert concat_cat.hc_structure.catalog_path is None

    # Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(left_cat, right_cat)


def test_concat_catalog_row_content(small_sky_order1_catalog):
    """Verify row content consistency after concatenation.

    Every row in the concatenated catalog should exactly match the corresponding
    row (by 'id') from either the left or right catalog. All column values must
    remain identical.

    Args:
        small_sky_order1_catalog: A small test catalog fixture.
    """
    cone1 = ConeSearch(325, -55, 36000)
    cone2 = ConeSearch(325, -25, 36000)

    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    concat_cat = left_cat.concat(right_cat)

    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    for _, row in df_concat.iterrows():
        row_id = row["id"]
        match_left = df_left[df_left["id"] == row_id]
        match_right = df_right[df_right["id"] == row_id]
        assert not (match_left.empty and match_right.empty), f"id {row_id} not found in left nor right"
        expected_row = match_left.iloc[0] if not match_left.empty else match_right.iloc[0]
        for col in df_concat.columns:
            assert row[col] == expected_row[col], f"Different value in column '{col}' for id {row_id}"

    # Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(left_cat, right_cat)


def test_concat_catalog_margin_content(small_sky_order1_collection_catalog):
    """Verify row content consistency in margins after concatenation.

    Every row in the concatenated catalog's margin must exactly match the
    corresponding row (by 'id') in either the left or right catalog's margin.
    All column values must remain identical.

    Args:
        small_sky_order1_collection_catalog: A collection catalog fixture with margins.
    """
    cone1 = ConeSearch(325, -55, 36000)
    cone2 = ConeSearch(325, -25, 36000)

    left_cat = small_sky_order1_collection_catalog.search(cone1)
    right_cat = small_sky_order1_collection_catalog.search(cone2)

    concat_cat = left_cat.concat(right_cat)

    margin_left_obj = getattr(left_cat, "margin", None)
    margin_right_obj = getattr(right_cat, "margin", None)
    margin_concat_obj = getattr(concat_cat, "margin", None)
    assert margin_left_obj is not None and margin_right_obj is not None and margin_concat_obj is not None

    margin_left = margin_left_obj.compute()
    margin_right = margin_right_obj.compute()
    margin_concat = margin_concat_obj.compute()

    for _, row in margin_concat.iterrows():
        row_id = row["id"]
        match_left = margin_left[margin_left["id"] == row_id]
        match_right = margin_right[margin_right["id"] == row_id]
        assert not (match_left.empty and match_right.empty), f"id {row_id} not found in left nor right margin"
        expected_row = match_left.iloc[0] if not match_left.empty else match_right.iloc[0]
        for col in margin_concat.columns:
            assert (
                row[col] == expected_row[col]
            ), f"Different value in column '{col}' for id {row_id} in margin"

    # Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(left_cat, right_cat)


def test_concat_catalogs_with_different_schemas(small_sky_order1_collection_dir, test_data_dir, helpers):
    """Concatenate catalogs with different schemas and validate behavior.

    Validates that concatenating two catalogs with different column sets
    behaves correctly:
      1. Row count equals the sum of inputs.
      2. Output columns equal the union of input columns.
      3. Side-specific columns are NaN for rows originating from the other side.
      4. Common columns preserve their values.
      5. Concatenation is symmetric between left and right.

    Args:
        small_sky_order1_collection_dir: Directory for the left catalog fixture.
        test_data_dir: Base directory containing right catalog and margin cache.
        helpers: Utility fixture with helper assertions.
    """
    left_cat = cast(Catalog, lsdb.open_catalog(small_sky_order1_collection_dir))
    left_margin = getattr(left_cat, "margin", None)
    assert left_margin is not None

    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"
    right_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    right_margin = getattr(right_cat, "margin", None)
    assert right_margin is not None

    left_df = left_cat.compute()
    right_df = right_cat.compute()

    assert "id" in left_df.columns, "Expected 'id' column in left_df"
    assert "source_id" in right_df.columns, "Expected 'source_id' column in right_df"

    concat_cat = left_cat.concat(right_cat)
    concat_df = concat_cat.compute()

    # (1) Row count
    assert len(concat_df) == len(left_df) + len(
        right_df
    ), f"Expected {len(left_df) + len(right_df)} rows, got {len(concat_df)}"

    # (2) Columns = union
    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)
    expected_cols = sorted(left_cols | right_cols)
    assert (
        sorted(concat_df.columns.tolist()) == expected_cols
    ), "Concatenated columns differ from the union of input columns"

    # Identify origin by presence of side-specific IDs
    mask_left_rows = concat_df["id"].notna()
    mask_right_rows = concat_df["source_id"].notna()

    # (3a) LEFT-only columns
    left_only_cols = left_cols - right_cols
    for col in left_only_cols:
        assert (
            concat_df.loc[mask_right_rows, col].isna().all()
        ), f"Column '{col}' should be NaN on RIGHT-origin rows"
        pd.testing.assert_series_equal(
            concat_df.loc[mask_left_rows, col].sort_index(),
            left_df[col].sort_index(),
            check_names=False,
            check_dtype=False,
        )

    # (3b) RIGHT-only columns
    right_only_cols = right_cols - left_cols
    for col in right_only_cols:
        assert (
            concat_df.loc[mask_left_rows, col].isna().all()
        ), f"Column '{col}' should be NaN on LEFT-origin rows"
        pd.testing.assert_series_equal(
            concat_df.loc[mask_right_rows, col].sort_index(),
            right_df[col].sort_index(),
            check_names=False,
            check_dtype=False,
        )

    # (4) Common columns
    common_cols = left_cols & right_cols
    for col in common_cols:
        pd.testing.assert_series_equal(
            concat_df.loc[mask_left_rows, col].sort_index(),
            left_df[col].sort_index(),
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            concat_df.loc[mask_right_rows, col].sort_index(),
            right_df[col].sort_index(),
            check_names=False,
            check_dtype=False,
        )

    # Structural checks
    assert isinstance(concat_cat._ddf, nd.NestedFrame)
    helpers.assert_divisions_are_correct(concat_cat)

    # (5) Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(left_cat, right_cat)


@pytest.mark.parametrize("use_low_order", [True, False])
def test_concat_margin_with_low_and_high_orders(use_low_order):
    """Validate margin concatenation when mixing high-order and low-order catalogs.

    This test builds two catalogs directly from DataFrames:
      * `cat_high` is always created at high order (order_high = 2).
      * `cat_low` is created either at a lower order (order_low = 1) when
        `use_low_order=True`, or at the same high order when `use_low_order=False`.

    In both cases:
      * The concatenated margin must contain exactly the three expected rows.
      * Concatenation must remain symmetric between left.concat(right)
        and right.concat(left).

    Args:
        use_low_order (bool): Parameter to test low-order vs. high-order input.
    """
    order_low = 1
    order_high = 2

    df_high_main = pd.DataFrame({"id": [1, 2], "ra": [314, 350], "dec": [-50, -70]})
    df_high_margin = pd.DataFrame({"id": [3, 4], "ra": [324.4, 324.4], "dec": [-60.099, -60.1]})
    df_low_main = pd.DataFrame({"id": [5, 6, 7, 8], "ra": [355, 275, 315, 315], "dec": [-70, -70, -80, -45]})
    df_low_into_df_high_margin = pd.DataFrame({"id": [9], "ra": [324.4], "dec": [-60.101]})

    df_high = pd.concat([df_high_main, df_high_margin], ignore_index=True)
    df_low = pd.concat([df_low_main, df_low_into_df_high_margin], ignore_index=True)

    cat_high = lsdb.from_dataframe(df_high, lowest_order=order_high, highest_order=order_high)
    if use_low_order:
        cat_low = lsdb.from_dataframe(df_low, lowest_order=order_low, highest_order=order_low)
    else:
        cat_low = lsdb.from_dataframe(df_low, lowest_order=order_high, highest_order=order_high)

    concat_cat = cat_high.concat(cat_low)
    margin_obj = getattr(concat_cat, "margin", None)
    assert margin_obj is not None
    margin_df = margin_obj.compute()

    expected_index = [
        3205067508097231189,  # id=4
        3205067511012692051,  # id=3
        3205067507930193833,  # id=9
    ]
    expected_rows = pd.DataFrame(
        {"id": [4, 3, 9], "ra": [324.4, 324.4, 324.4], "dec": [-60.1, -60.099, -60.101]},
        index=pd.Index(expected_index, name=SPATIAL_INDEX_COLUMN),
    ).sort_index()

    sub = margin_df.loc[expected_rows.index, ["id", "ra", "dec"]].sort_index()

    pd.testing.assert_frame_equal(
        sub.reset_index()[[SPATIAL_INDEX_COLUMN, "id", "ra", "dec"]],
        expected_rows.reset_index()[[SPATIAL_INDEX_COLUMN, "id", "ra", "dec"]],
        check_dtype=False,
        check_like=True,
    )

    only_expected = margin_df.index.isin(expected_rows.index)
    assert only_expected.all(), "Found margin indices beyond the three expected entries for this scenario."

    # Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(cat_high, cat_low)


@pytest.mark.parametrize("use_low_order", [True, False])
def test_concat_margin_with_different_schemas_and_orders(use_low_order):
    """Validate margin concatenation when catalogs have different schemas.

    This test creates two catalogs with different schemas:
      * `cat_high` has columns (id_1, ra_1, dec_1).
      * `cat_low` has columns (id_2, ra_2, dec_2).

    After concatenation:
      * The margin must contain exactly the three expected entries.
      * Columns that belong only to one schema must be null for rows
        originating from the other schema.
      * Concatenation must remain symmetric between left.concat(right)
        and right.concat(left).

    Args:
        use_low_order (bool): Parameter to test low-order vs. high-order input.
    """
    order_low = 1
    order_high = 2

    df_high_main = pd.DataFrame({"id_1": [1, 2], "ra_1": [314, 350], "dec_1": [-50, -70]})
    df_high_margin = pd.DataFrame({"id_1": [3, 4], "ra_1": [324.4, 324.4], "dec_1": [-60.099, -60.1]})
    df_low_main = pd.DataFrame(
        {"id_2": [5, 6, 7, 8], "ra_2": [355, 275, 315, 315], "dec_2": [-70, -70, -80, -45]}
    )
    df_low_into_df_high_margin = pd.DataFrame({"id_2": [9], "ra_2": [324.4], "dec_2": [-60.101]})

    df_high = pd.concat([df_high_main, df_high_margin], ignore_index=True)
    df_low = pd.concat([df_low_main, df_low_into_df_high_margin], ignore_index=True)

    cat_high = lsdb.from_dataframe(
        df_high, ra_column="ra_1", dec_column="dec_1", lowest_order=order_high, highest_order=order_high
    )
    if use_low_order:
        cat_low = lsdb.from_dataframe(
            df_low, ra_column="ra_2", dec_column="dec_2", lowest_order=order_low, highest_order=order_low
        )
    else:
        cat_low = lsdb.from_dataframe(
            df_low, ra_column="ra_2", dec_column="dec_2", lowest_order=order_high, highest_order=order_high
        )

    concat_cat = cat_high.concat(cat_low)
    margin_obj = getattr(concat_cat, "margin", None)
    assert margin_obj is not None
    margin_df = margin_obj.compute()

    expected_index = [
        3205067508097231189,  # id_1=4
        3205067511012692051,  # id_1=3
        3205067507930193833,  # id_2=9
    ]
    expected_rows = pd.DataFrame(
        {
            "id_1": [4, 3, pd.NA],
            "ra_1": [324.4, 324.4, pd.NA],
            "dec_1": [-60.1, -60.099, pd.NA],
            "id_2": [pd.NA, pd.NA, 9],
            "ra_2": [pd.NA, pd.NA, 324.4],
            "dec_2": [pd.NA, pd.NA, -60.101],
        },
        index=pd.Index(expected_index, name=SPATIAL_INDEX_COLUMN),
    ).sort_index()

    cols = ["id_1", "ra_1", "dec_1", "id_2", "ra_2", "dec_2"]
    got = margin_df.loc[expected_rows.index, cols].sort_index().reset_index()[[SPATIAL_INDEX_COLUMN] + cols]
    exp = expected_rows.reset_index()[[SPATIAL_INDEX_COLUMN] + cols]

    got_norm = _normalize_na(got)
    exp_norm = _normalize_na(exp)

    pd.testing.assert_frame_equal(got_norm, exp_norm, check_dtype=False, check_like=True)

    only_expected = margin_df.index.isin(expected_rows.index)
    assert only_expected.all(), "Found margin indices beyond the three expected entries for this scenario."

    # Symmetry check (main and margin handled internally)
    _assert_concat_symmetry(cat_high, cat_low)


def test_concat_warn_left_no_margin(test_data_dir):
    """Warn when only right catalog has margin, and result must not include margin.

    This test opens the same physical catalog on both sides:
      * Left side without a margin cache.
      * Right side with an explicit margin cache.

    Expected behavior:
      * A warning is raised.
      * The concatenated result does not carry a margin dataset.
      * Main table concatenation remains symmetric.

    Args:
        test_data_dir: Base directory containing test catalogs.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    left_cat = cast(Catalog, lsdb.open_catalog(right_dir))
    left_margin = getattr(left_cat, "margin", None)
    assert left_margin is None, "Left catalog unexpectedly has a margin"

    right_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    right_margin = getattr(right_cat, "margin", None)
    assert right_margin is not None, "Right catalog should have a margin"

    # The implementation now emits a generic message ("One side has no margin ...")
    with pytest.warns(UserWarning, match=r"One side has no margin"):
        concat_cat = left_cat.concat(right_cat)
        concat_margin = getattr(concat_cat, "margin", None)

    assert concat_margin is None, "Concatenated catalog should not include margin when only one side has it"

    _assert_concat_symmetry(left_cat, right_cat)


def test_concat_warn_right_no_margin(test_data_dir):
    """Warn when only left catalog has margin, and result must not include margin.

    This test flips the setup:
      * Left side with an explicit margin cache.
      * Right side without margin.

    Expected behavior:
      * A warning is raised.
      * The concatenated result does not carry a margin dataset.
      * Main table concatenation remains symmetric.

    Args:
        test_data_dir: Base directory containing test catalogs.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    left_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    left_margin = getattr(left_cat, "margin", None)
    assert left_margin is not None, "Left catalog should have a margin"

    right_cat = cast(Catalog, lsdb.open_catalog(right_dir))
    right_margin = getattr(right_cat, "margin", None)
    assert right_margin is None, "Right catalog unexpectedly has a margin"

    with pytest.warns(UserWarning, match=r"One side has no margin"):
        concat_cat = left_cat.concat(right_cat)
        concat_margin = getattr(concat_cat, "margin", None)

    assert concat_margin is None, "Concatenated catalog should not include margin when only one side has it"

    _assert_concat_symmetry(left_cat, right_cat)


def test_concat_kwargs_forwarding_does_not_change_content(test_data_dir):
    """Verify that kwargs passed to Catalog.concat do not change logical content.

    This test ensures that extra kwargs such as `ignore_index=True` are accepted
    by Catalog.concat, but do not alter the logical content of the concatenated
    result.

    Args:
        test_data_dir: Base directory containing test catalogs.
    """
    src_dir = test_data_dir / "small_sky_order3_source"
    cat = cast(Catalog, lsdb.open_catalog(src_dir))

    left = cat.search(ConeSearch(325, -55, 36000))
    right = cat.search(ConeSearch(325, -25, 36000))

    concat_default = left.concat(right)
    concat_kwargs = left.concat(right, ignore_index=True)

    df_default = concat_default.compute().reset_index()
    df_kwargs = concat_kwargs.compute().reset_index()
    df_default, df_kwargs = _align_columns(df_default, df_kwargs)
    assert _row_multiset(df_default) == _row_multiset(
        df_kwargs
    ), "Passing kwargs to concat should not change the logical content"


def test_concat_both_margins_uses_smallest_threshold(small_sky_order1_collection_dir, test_data_dir):
    """Verify margin threshold selection when both sides have margins.

    When both catalogs have a margin, the concatenated margin must use the
    smallest of the two input thresholds.

    Args:
        small_sky_order1_collection_dir: Directory containing a collection with margin.
        test_data_dir: Base directory containing another catalog and margin cache.
    """
    left_cat = cast(Catalog, lsdb.open_catalog(small_sky_order1_collection_dir))
    left_margin = getattr(left_cat, "margin", None)
    assert left_margin is not None, "Left catalog should have a margin"
    left_thr = left_margin.hc_structure.catalog_info.margin_threshold

    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"
    right_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    right_margin = getattr(right_cat, "margin", None)
    assert right_margin is not None, "Right catalog should have a margin"
    right_thr = right_margin.hc_structure.catalog_info.margin_threshold

    assert left_thr is not None and right_thr is not None, "Input margin thresholds must be defined"

    concat_cat = left_cat.concat(right_cat)
    concat_margin = getattr(concat_cat, "margin", None)

    assert concat_margin is not None, "Concatenated catalog should include a margin when both sides have one"
    got_thr = concat_margin.hc_structure.catalog_info.margin_threshold
    exp_thr = min(left_thr, right_thr)
    assert float(got_thr) == float(exp_thr), f"Expected margin_threshold={exp_thr} but got {got_thr}"

    _assert_concat_symmetry(left_cat, right_cat)


@pytest.mark.parametrize("na_sentinel", ["pd.NA", "np.nan"])
def test_concat_preserves_all_na_columns(test_data_dir, na_sentinel):
    """Ensure that all-NA columns are preserved during concatenation.

    Columns that are entirely null (all-NA) in both catalogs must remain present
    in the output and fully NaN after concatenation. Other content should remain
    unchanged compared to a simple vertical stack.

    Args:
        test_data_dir: Base directory containing test catalogs.
        na_sentinel (str): Type of null sentinel to test ("pd.NA" or "np.nan").
    """
    src_dir = test_data_dir / "small_sky_order3_source"
    left = cast(Catalog, lsdb.open_catalog(src_dir))
    right = cast(Catalog, lsdb.open_catalog(src_dir))

    na_value = pd.NA if na_sentinel == "pd.NA" else np.nan

    left_df = left.compute()
    right_df = right.compute()
    left_df["only_na"] = na_value
    right_df["only_na"] = na_value

    left2 = lsdb.from_dataframe(left_df, ra_column="source_ra", dec_column="source_dec")
    right2 = lsdb.from_dataframe(right_df, ra_column="source_ra", dec_column="source_dec")

    concat_cat = left2.concat(right2)
    concat_df = concat_cat.compute()

    assert "only_na" in concat_df.columns
    assert concat_df["only_na"].isna().all()

    expected_stack = pd.concat(
        [left_df.drop(columns=["only_na"]), right_df.drop(columns=["only_na"])],
        ignore_index=True,
    )
    got_no_onlyna = concat_df.drop(columns=["only_na"]).reset_index(drop=True)

    exp_aligned, got_aligned = _align_columns(expected_stack, got_no_onlyna)
    assert _row_multiset(exp_aligned) == _row_multiset(got_aligned), (
        f"Concat content should match vertical stack even with an all-NA column "
        f"(na_sentinel={na_sentinel})"
    )


def test__is_all_na_returns_true_for_size_zero():
    """Verify _is_all_na correctness for empty, all-null, and mixed DataFrames."""
    df_empty = pd.DataFrame(columns=["a", "b"]).iloc[0:0]
    assert df_empty.size == 0
    assert concat_catalog_data._is_all_na(df_empty) is True

    df_all_na = pd.DataFrame({"a": [np.nan, np.nan], "b": [pd.NA, pd.NA]})
    assert concat_catalog_data._is_all_na(df_all_na) is True

    df_some = pd.DataFrame({"a": [np.nan, 1.0]})
    assert concat_catalog_data._is_all_na(df_some) is False


def test__concat_meta_safe_skips_none_and_all_na_parts():
    """Verify _concat_meta_safe skips None, empty, and all-NA parts."""
    meta = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="float64")}).iloc[0:0]

    p_none = None
    p_empty = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="float64")})
    p_all_na = pd.DataFrame({"a": [np.nan, np.nan], "b": [pd.NA, pd.NA]})
    p_data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    out = concat_catalog_data._concat_meta_safe(meta, [p_none, p_empty, p_all_na, p_data], ignore_index=True)

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        p_data.reindex(columns=meta.columns).reset_index(drop=True),
        check_dtype=False,
    )


def test_filter_by_spatial_index_to_margin_raises_when_margin_order_smaller_than_order(
    monkeypatch,
):
    """Raise ValueError when derived margin_order < order.

    This test monkeypatches `hp.margin2order` to force an invalid result where
    `margin_order` is smaller than `order`, ensuring the error branch is hit.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    df = pd.DataFrame({"x": [1]}, index=pd.Index([0], name=merge_catalog_functions.SPATIAL_INDEX_COLUMN))

    monkeypatch.setattr(merge_catalog_functions.hp, "margin2order", lambda arr: np.array([0], dtype=int))

    with pytest.raises(
        ValueError, match=r"Margin order .* is smaller than the order .* Cannot generate margin"
    ):
        merge_catalog_functions.filter_by_spatial_index_to_margin(
            dataframe=df,
            order=1,
            pixel=0,
            margin_radius=0.1,
        )


def test_get_aligned_pixels_from_alignment_empty_mapping_returns_empty_list():
    """Return an empty list if alignment.pixel_mapping is empty."""
    dummy_alignment = SimpleNamespace(pixel_mapping=pd.DataFrame())
    got = merge_catalog_functions.get_aligned_pixels_from_alignment(dummy_alignment)
    assert not got
    assert isinstance(got, list)


def test__reindex_and_coerce_dtypes_success():
    """Verify _reindex_and_coerce_dtypes correctly coerces column dtypes.

    The function must:
      * Reindex columns to match meta.
      * Coerce values to the target dtype (e.g., float -> int, int -> float).
      * Convert object/string-like columns consistently to string dtype.
    """
    meta = pd.DataFrame(
        {
            "a": pd.Series(dtype="int64"),
            "b": pd.Series(dtype="float64"),
            "c": pd.Series(dtype="string"),
        }
    ).iloc[0:0]

    df = pd.DataFrame(
        {
            "b": [1, 2],  # should become float64
            "a": [3.0, 4.0],  # float -> int64
            "c": ["x", None],  # object -> string (None -> <NA>)
        }
    )

    out = concat_catalog_data._reindex_and_coerce_dtypes(df, meta)

    # Explicitly build expected with pd.NA for string dtype consistency
    expected = pd.DataFrame(
        {
            "a": pd.Series([3, 4], dtype="int64"),
            "b": pd.Series([1.0, 2.0], dtype="float64"),
            "c": pd.Series(["x", pd.NA], dtype="string"),
        }
    )

    assert list(out.columns) == ["a", "b", "c"]
    assert str(out["a"].dtype) == "int64"
    assert str(out["b"].dtype) == "float64"
    assert str(out["c"].dtype).startswith("string")

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected, check_dtype=False)


def test__reindex_and_coerce_dtypes_raises_typeerror_on_incompatible():
    """Raise TypeError when dtype coercion is impossible."""
    meta = pd.DataFrame({"x": pd.Series(dtype="int64")}).iloc[0:0]
    df = pd.DataFrame({"x": ["abc", "42"]})

    with pytest.raises(TypeError, match="Could not convert column 'x'"):
        concat_catalog_data._reindex_and_coerce_dtypes(df, meta)


def test__check_strict_column_types_raises_on_conflict():
    """Verify that _check_strict_column_types raises when dtypes conflict."""
    meta1 = pd.DataFrame({"x": pd.Series(dtype="int64")}).iloc[0:0]
    meta2 = pd.DataFrame({"x": pd.Series(dtype="float64")}).iloc[0:0]

    with pytest.raises(TypeError, match="Column 'x' has conflicting dtypes"):
        concat_catalog_data._check_strict_column_types(meta1, meta2)


# ---------------------------- extra helpers -------------------------------- #


def _make_plain_catalog_at_order(src_cat: Catalog, order: int) -> Catalog:
    """Materialize a no-margin catalog at a specific HEALPix order.

    Builds a catalog from the rows of `src_cat` but ensures no margin is created
    by passing `margin_threshold=None`. Uses the same RA/DEC column names as
    `src_cat`.

    Args:
        src_cat (Catalog): Source catalog to copy rows from.
        order (int): HEALPix order for both lowest and highest orders.

    Returns:
        Catalog: A new catalog at the requested order with no margin.
    """
    info = src_cat.hc_structure.catalog_info
    ra_col = info.ra_column or "ra"
    dec_col = info.dec_column or "dec"

    df = src_cat.compute().reset_index(drop=True)

    # Keep non-HIVE columns (and keep RA/DEC for correct spatial indexing).
    cols = [c for c in df.columns if c in {ra_col, dec_col} or not c.startswith("HIVE_")]
    df2 = df[cols].copy()

    return lsdb.from_dataframe(
        df2,
        ra_column=ra_col,
        dec_column=dec_col,
        lowest_order=order,
        highest_order=order,
        margin_threshold=None,  # critical to avoid accidental margin creation
        # margin_order remains default (-1); no margin will be generated.
    )


# ------------------------------ new tests ---------------------------------- #


def test_concat_ignore_empty_margins_left_missing_keeps_right_margin(test_data_dir):
    """Keep right margin when left lacks margin and ignore_empty_margins=True.

    The concatenation must:
      * Emit a warning indicating the missing side is treated as an empty margin.
      * Produce a margin dataset in the result.
      * Use the threshold from the right side's margin.
      * Contain only rows that are a subset of the right margin's rows.
      * Remain symmetric in content under the same kwargs.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    left_cat = cast(Catalog, lsdb.open_catalog(right_dir))
    assert getattr(left_cat, "margin", None) is None

    right_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    right_margin = getattr(right_cat, "margin", None)
    assert right_margin is not None
    rm = right_margin
    right_thr = rm.hc_structure.catalog_info.margin_threshold
    assert right_thr is not None

    with pytest.warns(UserWarning, match=r"ignore_empty_margins.*treated as empty|treated as empty"):
        concat_cat = left_cat.concat(right_cat, ignore_empty_margins=True)

    concat_margin = getattr(concat_cat, "margin", None)
    assert concat_margin is not None, "Expected concatenated catalog to carry a margin"
    got_thr = concat_margin.hc_structure.catalog_info.margin_threshold
    assert float(got_thr) == float(right_thr), f"Expected margin_threshold={right_thr}, got {got_thr}"

    # Subset check: concatenated margin rows must all exist in the right margin
    right_margin_df = right_margin.compute().reset_index()
    concat_margin_df = concat_margin.compute().reset_index()
    concat_margin_df_al, right_margin_df_al = _align_columns(concat_margin_df, right_margin_df)

    ms_concat = _row_multiset(concat_margin_df_al)
    ms_right = _row_multiset(right_margin_df_al)
    for row, cnt in ms_concat.items():
        assert cnt <= ms_right[row], "Concatenated margin contains rows not present in RIGHT's margin"

    # Symmetry with forwarded kwargs
    _assert_concat_symmetry(left_cat, right_cat, ignore_empty_margins=True)


def test_concat_ignore_empty_margins_right_missing_keeps_left_margin(test_data_dir):
    """Keep left margin when right lacks margin and ignore_empty_margins=True.

    The concatenation must:
      * Emit a warning indicating the missing side is treated as an empty margin.
      * Produce a margin dataset in the result.
      * Use the threshold from the left side's margin.
      * Contain only rows that are a subset of the left margin's rows.
      * Remain symmetric in content under the same kwargs.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    left_cat = cast(Catalog, lsdb.open_catalog(right_dir, margin_cache=right_margin_dir))
    left_margin = getattr(left_cat, "margin", None)
    assert left_margin is not None
    lm = left_margin
    left_thr = lm.hc_structure.catalog_info.margin_threshold
    assert left_thr is not None

    right_cat = cast(Catalog, lsdb.open_catalog(right_dir))
    assert getattr(right_cat, "margin", None) is None

    with pytest.warns(UserWarning, match=r"ignore_empty_margins.*treated as empty|treated as empty"):
        concat_cat = left_cat.concat(right_cat, ignore_empty_margins=True)

    concat_margin = getattr(concat_cat, "margin", None)
    assert concat_margin is not None, "Expected concatenated catalog to carry a margin"
    got_thr = concat_margin.hc_structure.catalog_info.margin_threshold
    assert float(got_thr) == float(left_thr), f"Expected margin_threshold={left_thr}, got {got_thr}"

    # Subset check: concatenated margin rows must all exist in the left margin
    left_margin_df = left_margin.compute().reset_index()
    concat_margin_df = concat_margin.compute().reset_index()
    concat_margin_df_al, left_margin_df_al = _align_columns(concat_margin_df, left_margin_df)

    ms_concat = _row_multiset(concat_margin_df_al)
    ms_left = _row_multiset(left_margin_df_al)
    for row, cnt in ms_concat.items():
        assert cnt <= ms_left[row], "Concatenated margin contains rows not present in LEFT's margin"

    # Symmetry with forwarded kwargs
    _assert_concat_symmetry(left_cat, right_cat, ignore_empty_margins=True)


@pytest.mark.parametrize(
    "left_kind,right_kind",
    [
        # 1) left lower order (no margin), right higher order (with margin)
        ("low_no_margin", "high_with_margin"),
        # 2) left higher order (no margin), right lower order (with margin)
        ("high_no_margin", "low_with_margin"),
        # 3) left lower order (with margin), right higher order (no margin)
        ("low_with_margin", "high_no_margin"),
        # 4) left higher order (with margin), right lower order (no margin)
        ("high_with_margin", "low_no_margin"),
    ],
)
def test_concat_ignore_empty_margins_mixed_orders(
    left_kind: str,
    right_kind: str,
    small_sky_order1_collection_dir,
    test_data_dir,
):
    """Validate ignore_empty_margins across mixed HEALPix orders.

    For each scenario, exactly one side provides a margin and the other is
    treated as an empty margin at the same radius. We expect:
      * A warning about treating the missing side as empty.
      * A concatenated margin to be present with threshold equal to the
        threshold of the side that had a margin.
      * Symmetry under the same kwargs.

    Notes:
        We do NOT assert that the concatenated margin is a subset of the
        base margin. With mixed orders, LSDB may legitimately include rows
        that come from the other side’s partition after spatial filtering
        at the aligned order/margin ring.

    Args:
        left_kind (str): Selector for the left catalog kind.
        right_kind (str): Selector for the right catalog kind.
        small_sky_order1_collection_dir: Fixture path for a low-order catalog with margin.
        test_data_dir: Base directory containing a high-order catalog and its margin cache.
    """
    # Base catalogs WITH margins
    low_with = cast(Catalog, lsdb.open_catalog(small_sky_order1_collection_dir))
    low_margin = getattr(low_with, "margin", None)
    assert low_margin is not None
    low_thr = low_margin.hc_structure.catalog_info.margin_threshold
    assert low_thr is not None

    hi_dir = test_data_dir / "small_sky_order3_source"
    hi_margin_dir = test_data_dir / "small_sky_order3_source_margin"
    high_with = cast(Catalog, lsdb.open_catalog(hi_dir, margin_cache=hi_margin_dir))
    high_margin = getattr(high_with, "margin", None)
    assert high_margin is not None
    high_thr = high_margin.hc_structure.catalog_info.margin_threshold
    assert high_thr is not None

    # NO-MARGIN variants at the desired orders (fixtures are known: low=1, high=3)
    low_order = 1
    low_no = _make_plain_catalog_at_order(low_with, low_order)
    assert getattr(low_no, "margin", None) is None

    # High-order, no margin: open without margin_cache (already order=3)
    high_no = cast(Catalog, lsdb.open_catalog(hi_dir))
    assert getattr(high_no, "margin", None) is None

    def pick_catalog(kind: str) -> Catalog:
        if kind == "low_with_margin":
            return low_with
        if kind == "high_with_margin":
            return high_with
        if kind == "low_no_margin":
            return low_no
        if kind == "high_no_margin":
            return high_no
        raise ValueError(kind)

    left = pick_catalog(left_kind)
    right = pick_catalog(right_kind)

    # Sanity: exactly one side must have a margin in each scenario
    left_m = getattr(left, "margin", None)
    right_m = getattr(right, "margin", None)
    assert (left_m is None) ^ (right_m is None), "Test setup error: expected exactly one side with margin."

    # Perform concat with ignore_empty_margins=True
    with pytest.warns(UserWarning, match=r"ignore_empty_margins.*treated as empty|treated as empty"):
        concatenated = left.concat(right, ignore_empty_margins=True)

    # A margin must be present
    conc_margin = getattr(concatenated, "margin", None)
    assert conc_margin is not None, f"Expected a margin for case {left_kind} + {right_kind}"

    # Threshold must match the side that had a margin
    got_thr = float(conc_margin.hc_structure.catalog_info.margin_threshold)
    base_m = left_m if left_m is not None else right_m
    assert base_m is not None
    exp_thr = float(base_m.hc_structure.catalog_info.margin_threshold)

    assert (
        got_thr == exp_thr
    ), f"[{left_kind} + {right_kind}] expected margin_threshold={exp_thr}, got {got_thr}"

    # Symmetry with forwarded kwargs
    _assert_concat_symmetry(left, right, ignore_empty_margins=True)


# --- New lightweight unit tests for handle_margins_for_concat ---


class _DummyHCCatalog:  # pylint: disable=too-few-public-methods
    """Minimal stub to mimic a HATS catalog-like structure."""

    def __init__(self, catalog_info, pixel_tree=None):
        self.catalog_info = catalog_info
        self.pixel_tree = pixel_tree


class _DummyMargin:  # pylint: disable=too-few-public-methods
    """Margin stub with just enough surface for the function under test."""

    def __init__(self, radius: float):
        # catalog_info with margin_threshold
        self.hc_structure = _DummyHCCatalog(catalog_info=SimpleNamespace(margin_threshold=radius))
        # capture arguments passed to _create_updated_dataset
        self._last_kwargs = None

    def _create_updated_dataset(self, *, ddf, ddf_pixel_map, hc_structure, updated_catalog_info_params):
        # store for assertions
        self._last_kwargs = {
            "ddf": ddf,
            "ddf_pixel_map": ddf_pixel_map,
            "hc_structure": hc_structure,
            "updated_catalog_info_params": updated_catalog_info_params,
        }
        # return a sentinel we can assert on
        return ("UPDATED", hc_structure, updated_catalog_info_params)


def test_handle_margins_both_have_margin_uses_min_threshold_and_calls_concat(monkeypatch):
    """
    When both sides have margins, the function must:
      * use the smallest of the two thresholds,
      * call concat_margin_data with that radius,
      * return the result of _create_updated_dataset from the left margin.
    """
    left_margin = _DummyMargin(radius=1.5)
    right_margin = _DummyMargin(radius=2.5)

    left = SimpleNamespace(margin=left_margin)
    right = SimpleNamespace(margin=right_margin)

    # Spy concat_margin_data to capture args and return a lightweight triple
    called = {}

    def _fake_concat_margin_data(left_arg, right_arg, radius, **kwargs):
        called["args"] = (left_arg, right_arg, radius, kwargs)
        # mimic (ddf, ddf_map, alignment) where alignment has pixel_tree
        return ("DD", "MAP", SimpleNamespace(pixel_tree="PIXELS"))

    monkeypatch.setattr(concat_catalog_data, "concat_margin_data", _fake_concat_margin_data)

    got = concat_catalog_data.handle_margins_for_concat(left, right, ignore_empty_margins=False)

    # Check concat was called with the min radius (1.5)
    assert "args" in called, "concat_margin_data was not called"
    _left_arg, _right_arg, radius_used, _kwargs = called["args"]
    assert pytest.approx(radius_used, rel=0, abs=0) == 1.5

    # The return should be the sentinel from _create_updated_dataset
    assert isinstance(got, tuple) and got[0] == "UPDATED"

    # And the updated_catalog_info_params should carry the chosen radius
    updated_params = got[2]
    assert isinstance(updated_params, dict)
    assert pytest.approx(updated_params.get("margin_threshold"), rel=0, abs=0) == 1.5

    # Sanity: left margin recorded the kwargs we passed into _create_updated_dataset
    assert left_margin._last_kwargs is not None
    assert left_margin._last_kwargs["hc_structure"].catalog_info is left_margin.hc_structure.catalog_info
    assert left_margin._last_kwargs["updated_catalog_info_params"]["margin_threshold"] == 1.5


def test_handle_margins_neither_side_has_margin_returns_none_without_warning():
    """
    When neither side has a margin, the function must simply return None
    and emit no warnings.
    """
    left = SimpleNamespace(margin=None)
    right = SimpleNamespace(margin=None)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        got = concat_catalog_data.handle_margins_for_concat(left, right, ignore_empty_margins=False)

    assert got is None


def test_handle_margins_one_side_has_margin_legacy_warns_and_returns_none():
    """
    When exactly one side has a margin and ignore_empty_margins=False:
      * a warning must be emitted, and
      * the function must return None (legacy behavior).
    """
    left_margin = _DummyMargin(radius=1.0)
    left = SimpleNamespace(margin=left_margin)
    right = SimpleNamespace(margin=None)

    with pytest.warns(UserWarning, match=r"One side has no margin"):
        got = concat_catalog_data.handle_margins_for_concat(left, right, ignore_empty_margins=False)

    assert got is None


def test_handle_margins_one_side_has_margin_keep_existing_calls_concat(monkeypatch):
    """
    When exactly one side has a margin and ignore_empty_margins=True:
      * concat_margin_data must be called with the existing radius, and
      * the result must come from existing._create_updated_dataset.
    """
    existing = _DummyMargin(radius=2.0)
    left = SimpleNamespace(margin=existing)
    right = SimpleNamespace(margin=None)

    called = {}

    def _fake_concat_margin_data(left_arg, right_arg, radius, **kwargs):
        called["args"] = (left_arg, right_arg, radius, kwargs)
        return ("DD", "MAP", SimpleNamespace(pixel_tree="PIXELS"))

    monkeypatch.setattr(concat_catalog_data, "concat_margin_data", _fake_concat_margin_data)

    with pytest.warns(UserWarning, match=r"ignore_empty_margins=True.*treated as empty"):
        got = concat_catalog_data.handle_margins_for_concat(left, right, ignore_empty_margins=True)

    # Check that it was called with the existing radius
    assert "args" in called
    _l, _r, radius_used, _kw = called["args"]
    assert radius_used == 2.0

    # Check return from _create_updated_dataset
    assert isinstance(got, tuple) and got[0] == "UPDATED"
    assert got[2]["margin_threshold"] == 2.0
