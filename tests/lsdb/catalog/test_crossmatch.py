from typing import Tuple

import pandas as pd
import hipscat as hc
import pytest


def test_kdtree_crossmatch(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog).compute()
    assert len(xmatched) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_kdtree_crossmatch_thresh(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005):
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, d_thresh=0.005).compute()
    assert len(xmatched) == len(xmatch_correct_005)
    for _, correct_row in xmatch_correct_005.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_kdtree_crossmatch_multiple_neighbors(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t_no_margin):
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, n_neighbors=3, d_thresh=2).compute()
    assert len(xmatched) == len(xmatch_correct_3n_2t_no_margin)
    for _, correct_row in xmatch_correct_3n_2t_no_margin.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[(xmatched["id_small_sky"] == correct_row["ss_id"]) & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])]
        assert len(xmatch_row) == 1
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_custom_crossmatch_algorithm(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, algorithm=mock_crossmatch_algorithm, mock_results=xmatch_mock).compute()
    assert len(xmatched) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def mock_crossmatch_algorithm(
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_order: int,
        left_pixel: int,
        right_order: int,
        right_pixel: int,
        left_metadata: hc.catalog.Catalog,
        right_metadata: hc.catalog.Catalog,
        suffixes: Tuple[str, str],
        mock_results: pd.DataFrame = None):
    left = left.copy(deep=False)
    right = right.copy(deep=False)
    left_reset = left.reset_index(drop=True)
    right_reset = right.reset_index(drop=True)
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left.rename(columns=left_columns_renamed, inplace=True)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right.rename(columns=right_columns_renamed, inplace=True)
    mock_results = mock_results[mock_results["ss_id"].isin(left_reset["id"].values)]
    left_indexes = mock_results.apply(lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1)
    right_indexes = mock_results.apply(lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1)
    left_join_part = left.iloc[left_indexes.values].reset_index()
    right_join_part = right.iloc[right_indexes.values].reset_index(drop=True)
    out = pd.concat(
        [
            left_join_part,  # select the rows of the left table
            right_join_part  # select the rows of the right table
        ], axis=1)
    out.set_index("_hipscat_index", inplace=True)
    out["_DIST"] = mock_results["dist"].to_numpy()

    return out
