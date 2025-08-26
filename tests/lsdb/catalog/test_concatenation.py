from collections import Counter

import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb
import lsdb.nested as nd
from lsdb import ConeSearch

# ------------------------------- helpers ---------------------------------- #


def _normalize_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any null sentinel (NaN/pd.NA/None) to np.nan and cast columns to 'object'
    so assert_frame_equal compares nulls consistently regardless of dtype/backing.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype("object")
        out[c] = out[c].where(pd.notna(out[c]), np.nan)
    return out


def _row_multiset(df: pd.DataFrame) -> Counter:
    """
    Convert a DataFrame into a multiset (Counter) of row-tuples, ignoring
    row order but preserving multiplicity. Columns must already be aligned.
    - We canonicalize values so that 1 == 1.0 and all nulls -> None.
    - Columns are sorted by name to ignore column ordering differences.
    """
    cols = sorted(df.columns)

    def _canon(v):
        if pd.isna(v):
            return None
        # Treat numeric values uniformly as float to avoid 1 vs 1.0 mismatches
        if isinstance(v, (int, np.integer, float, np.floating, np.number, bool, np.bool_)):
            return float(v)
        return v

    rows = (tuple(_canon(x) for x in row) for row in df[cols].itertuples(index=False, name=None))
    return Counter(rows)


def _align_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two frames to the union of their columns (sorted for determinism),
    filling missing columns with NaN.
    """
    all_cols = sorted(set(df1.columns) | set(df2.columns))
    return df1.reindex(columns=all_cols), df2.reindex(columns=all_cols)


def _assert_concat_symmetry(
    left_cat, right_cat, use_margin: bool = False, cols_subset: list[str] | None = None
):
    """
    Assert that left.concat(right) and right.concat(left) contain the same content,
    allowing differences in row order and column order. Content includes the
    spatial-index (we compare on reset_index()).

    Parameters
    ----------
    use_margin : bool
        If True, compare the .margin DataFrames; otherwise compare the main tables.
    cols_subset : list[str] | None
        If provided, restrict the comparison to this subset of columns (after reset_index()).
    """
    lr = left_cat.concat(right_cat)
    rl = right_cat.concat(left_cat)

    df_lr = (lr.margin.compute() if use_margin else lr.compute()).reset_index()
    df_rl = (rl.margin.compute() if use_margin else rl.compute()).reset_index()

    # Optionally select a subset of columns (plus spatial index if present)
    if cols_subset is not None:
        keep = [c for c in [SPATIAL_INDEX_COLUMN] + cols_subset if c in df_lr.columns or c in df_rl.columns]
        df_lr = df_lr[keep]
        df_rl = df_rl[keep]

    # Align columns to the union and compare as multisets
    df_lr, df_rl = _align_columns(df_lr, df_rl)
    ms_lr = _row_multiset(df_lr)
    ms_rl = _row_multiset(df_rl)
    assert (
        ms_lr == ms_rl
    ), "left.concat(right) and right.concat(left) differ in content (as multisets of rows)."


# --------------------------------- tests ---------------------------------- #


def test_concat_catalog_row_count(small_sky_order1_catalog, helpers):
    """
    Concatenating two catalogs should produce a catalog whose row count equals
    the sum of the inputs, and the structure should be preserved.
    """
    # Two partially overlapping cones
    cone1 = ConeSearch(325, -55, 36000)  # 10 deg radius
    cone2 = ConeSearch(325, -25, 36000)  # 10 deg radius

    # Cone searches
    left_cat = small_sky_order1_catalog.search(cone1)
    right_cat = small_sky_order1_catalog.search(cone2)

    # Concatenate
    concat_cat = left_cat.concat(right_cat)

    # Compute
    df_left = left_cat.compute()
    df_right = right_cat.compute()
    df_concat = concat_cat.compute()

    # Row count = sum
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

    # Symmetry check: left.concat(right) vs right.concat(left)
    _assert_concat_symmetry(left_cat, right_cat, use_margin=False)


def test_concat_catalog_row_content(small_sky_order1_catalog):
    """
    Every row in the concatenated catalog should exactly match the corresponding
    row (by 'id') in either the left or the right catalog; all column values identical.
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

    # Symmetry: full-content equality as multisets
    _assert_concat_symmetry(left_cat, right_cat, use_margin=False)


def test_concat_catalog_margin_content(small_sky_order1_collection_catalog):
    """
    Every row in the concatenated catalog's margin should exactly match the corresponding
    row (by 'id') in either the left or right catalog's margin; all column values identical.
    Assumes same catalog structure and pixel order for this case.
    """
    cone1 = ConeSearch(325, -55, 36000)
    cone2 = ConeSearch(325, -25, 36000)

    left_cat = small_sky_order1_collection_catalog.search(cone1)
    right_cat = small_sky_order1_collection_catalog.search(cone2)

    concat_cat = left_cat.concat(right_cat)

    margin_left = left_cat.margin.compute()
    margin_right = right_cat.margin.compute()
    margin_concat = concat_cat.margin.compute()

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

    # Symmetry on margins
    _assert_concat_symmetry(left_cat, right_cat, use_margin=True)


def test_concat_catalogs_with_different_schemas(small_sky_order1_collection_dir, test_data_dir, helpers):
    """
    Concatenate two catalogs with different schemas (different column sets) and verify:
      1) Row count is the sum of inputs.
      2) Output columns are the union of input columns.
      3) Rows coming from LEFT (RIGHT) have RIGHT-only (LEFT-only) columns as NaN.
      4) For common columns, values match the originating side exactly.
      5) Symmetry: left.concat(right) and right.concat(left) have the same content
         (ignoring row/column ordering).
    """
    # LEFT: small_sky_order1_collection (collection)
    left_cat = lsdb.open_catalog(small_sky_order1_collection_dir)

    # RIGHT: small_sky_order3_source with margin cache
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"
    right_cat = lsdb.open_catalog(right_dir, margin_cache=right_margin_dir)

    # DataFrames
    left_df = left_cat.compute()
    right_df = right_cat.compute()

    assert "id" in left_df.columns, "Expected 'id' column in left_df"
    assert "source_id" in right_df.columns, "Expected 'source_id' column in right_df"

    # Concat
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

    # (3a) LEFT-only columns are NaN on RIGHT-origin rows; match on LEFT rows
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

    # (3b) RIGHT-only columns are NaN on LEFT-origin rows; match on RIGHT rows
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

    # (4) Common columns match on their originating side
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

    # (5) Symmetry on main tables (different schemas)
    _assert_concat_symmetry(left_cat, right_cat, use_margin=False)


@pytest.mark.parametrize("use_low_order", [True, False])
def test_concat_margin_with_low_and_high_orders(use_low_order):
    """
    Build two catalogs directly from DataFrames:
      - cat_high always at high order (order_high = 2)
      - cat_low at:
          * low order (order_low = 1) when use_low_order=True, or
          * the same high order when use_low_order=False

    In both cases, the concatenated margin must contain exactly the three expected rows.
    Also assert symmetry between left.concat(right) and right.concat(left) on margins.
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
    margin_df = concat_cat.margin.compute()  # pylint: disable=no-member

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

    # Symmetry on margins (use all columns available there)
    _assert_concat_symmetry(cat_high, cat_low, use_margin=True)


@pytest.mark.parametrize("use_low_order", [True, False])
def test_concat_margin_with_different_schemas_and_orders(use_low_order):
    """
    Create two catalogs with different schemas:
      - cat_high: columns id_1, ra_1, dec_1
      - cat_low : columns id_2, ra_2, dec_2
    Concatenate and assert the margin contains exactly the three expected entries,
    with nulls on the columns belonging to the other schema. Test two cases:
      * use_low_order=True  -> cat_low at low order (order_low=1)
      * use_low_order=False -> cat_low at the same high order as cat_high (order_high=2)
    Also assert symmetry between left.concat(right) and right.concat(left) on margins.
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
    margin_df = concat_cat.margin.compute()  # pylint: disable=no-member

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

    # Normalize nulls so NaN == pd.NA for comparison
    got_norm = _normalize_na(got)
    exp_norm = _normalize_na(exp)

    pd.testing.assert_frame_equal(got_norm, exp_norm, check_dtype=False, check_like=True)

    only_expected = margin_df.index.isin(expected_rows.index)
    assert only_expected.all(), "Found margin indices beyond the three expected entries for this scenario."

    # Symmetry on margins (different schemas)
    _assert_concat_symmetry(cat_high, cat_low, use_margin=True)


def test_concat_warn_left_no_margin(test_data_dir):
    """
    Left has NO margin; Right HAS margin -> must warn and result must NOT have margin.
    We use the same physical catalog on the right, but with an explicit margin_cache.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    # Left: open without margin_cache (no margin)
    left_cat = lsdb.open_catalog(right_dir)
    assert left_cat.margin is None, "Left catalog unexpectedly has a margin"

    # Right: open with margin_cache (has margin)
    right_cat = lsdb.open_catalog(right_dir, margin_cache=right_margin_dir)
    assert right_cat.margin is not None, "Right catalog should have a margin"

    # Expect a warning from concat since only the right side has margin
    with pytest.warns(UserWarning, match="Left catalog has no margin"):
        concat_cat = left_cat.concat(right_cat)

    # Result should not carry a margin dataset in this asymmetric case
    assert (
        concat_cat.margin is None
    ), "Concatenated catalog should not include margin when only one side has it"

    # Content symmetry (main tables) still holds
    _assert_concat_symmetry(left_cat, right_cat, use_margin=False)


def test_concat_warn_right_no_margin(test_data_dir):
    """
    Right has NO margin; Left HAS margin -> must warn and result must NOT have margin.
    We flip the previous setup: left gets a margin, right does not.
    """
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"

    # Left: with margin
    left_cat = lsdb.open_catalog(right_dir, margin_cache=right_margin_dir)
    assert left_cat.margin is not None, "Left catalog should have a margin"

    # Right: without margin
    right_cat = lsdb.open_catalog(right_dir)
    assert right_cat.margin is None, "Right catalog unexpectedly has a margin"

    with pytest.warns(UserWarning, match="Right catalog has no margin"):
        concat_cat = left_cat.concat(right_cat)

    assert (
        concat_cat.margin is None
    ), "Concatenated catalog should not include margin when only one side has it"

    # Content symmetry (main tables) still holds
    _assert_concat_symmetry(left_cat, right_cat, use_margin=False)


def test_concat_kwargs_forwarding_does_not_change_content(test_data_dir):
    """
    Ensure that **kwargs passed to Catalog.concat (e.g., ignore_index=True) are accepted
    and do not change the logical content. We compare the result with and without kwargs
    using the multiset-of-rows approach already used elsewhere.
    """
    # Use two partially overlapping cones from the same source to get non-trivial content
    src_dir = test_data_dir / "small_sky_order3_source"
    cat = lsdb.open_catalog(src_dir)

    left = cat.search(ConeSearch(325, -55, 36000))
    right = cat.search(ConeSearch(325, -25, 36000))

    concat_default = left.concat(right)  # no kwargs
    concat_kwargs = left.concat(right, ignore_index=True)  # exercise kwargs path

    # Compare content ignoring row/column order
    df_default = concat_default.compute().reset_index()
    df_kwargs = concat_kwargs.compute().reset_index()
    df_default, df_kwargs = _align_columns(df_default, df_kwargs)
    assert _row_multiset(df_default) == _row_multiset(
        df_kwargs
    ), "Passing kwargs to concat should not change the logical content"


def test_concat_both_margins_uses_smallest_threshold(small_sky_order1_collection_dir, test_data_dir):
    """
    When BOTH sides have a margin, the concatenated margin must be built using the
    SMALLEST of the two margin thresholds. We read thresholds from the inputs and
    check that the result's margin_info reflects min(left, right).
    """
    # LEFT: a collection that already ships with a margin cache
    left_cat = lsdb.open_catalog(small_sky_order1_collection_dir)
    assert left_cat.margin is not None, "Left catalog should have a margin"
    left_thr = left_cat.margin.hc_structure.catalog_info.margin_threshold

    # RIGHT: a source with an explicit margin_cache directory
    right_dir = test_data_dir / "small_sky_order3_source"
    right_margin_dir = test_data_dir / "small_sky_order3_source_margin"
    right_cat = lsdb.open_catalog(right_dir, margin_cache=right_margin_dir)
    assert right_cat.margin is not None, "Right catalog should have a margin"
    right_thr = right_cat.margin.hc_structure.catalog_info.margin_threshold

    # Sanity: thresholds must be numeric (some datasets store them as float)
    assert left_thr is not None and right_thr is not None, "Input margin thresholds must be defined"

    concat_cat = left_cat.concat(right_cat)

    # Result MUST have a margin and must use the smallest threshold
    assert (
        concat_cat.margin is not None
    ), "Concatenated catalog should include a margin when both sides have one"
    got_thr = concat_cat.margin.hc_structure.catalog_info.margin_threshold
    exp_thr = min(left_thr, right_thr)
    # Compare as floats to avoid dtype quirks
    assert float(got_thr) == float(exp_thr), f"Expected margin_threshold={exp_thr} but got {got_thr}"

    # Symmetry on margins still holds
    _assert_concat_symmetry(left_cat, right_cat, use_margin=True)


@pytest.mark.parametrize("na_sentinel", ["pd.NA", "np.nan"])
def test_concat_drops_all_na_cols_internally_but_reindexes_back(test_data_dir, na_sentinel):
    """
    Parametrized over the null sentinel used in the all-NA column ('pd.NA' or 'np.nan').

    If a column is present on both sides but is 100% null on all kept parts,
    the internal concat may drop it (to avoid pandas warnings) and then
    reindex it back to the meta schema. From the public API perspective,
    content must remain unchanged and the column must still exist (all NaN).
    """
    # Open two catalogs from the same source to ensure overlapping pixels
    src_dir = test_data_dir / "small_sky_order3_source"
    left = lsdb.open_catalog(src_dir)
    right = lsdb.open_catalog(src_dir)

    # Choose the sentinel
    na_value = pd.NA if na_sentinel == "pd.NA" else np.nan

    # Build DataFrames and add an all-null column using the chosen sentinel
    left_df = left.compute()
    right_df = right.compute()
    left_df["only_na"] = na_value
    right_df["only_na"] = na_value

    # Rebuild catalogs from the modified DataFrames
    # (explicit ra/dec to match this dataset's schema)
    left2 = lsdb.from_dataframe(left_df, ra_column="source_ra", dec_column="source_dec")
    right2 = lsdb.from_dataframe(right_df, ra_column="source_ra", dec_column="source_dec")

    # Concat and compute
    concat_cat = left2.concat(right2)
    concat_df = concat_cat.compute()

    # Column must exist and be entirely NaN after concat
    assert "only_na" in concat_df.columns, "Column 'only_na' should be reindexed back after internal drop"
    assert concat_df["only_na"].isna().all(), "Column 'only_na' must remain entirely NaN"

    # Logical content (ignoring the all-NA column) should equal simple vertical stack
    expected_stack = pd.concat(
        [left_df.drop(columns=["only_na"]), right_df.drop(columns=["only_na"])],
        ignore_index=True,
    )
    got_no_onlyna = concat_df.drop(columns=["only_na"]).reset_index(drop=True)

    # Align and compare as multisets (ignoring row/column order)
    exp_aligned, got_aligned = _align_columns(expected_stack, got_no_onlyna)
    assert _row_multiset(exp_aligned) == _row_multiset(
        got_aligned
    ), f"Concat content should match vertical stack even with an all-NA column (na_sentinel={na_sentinel})"
