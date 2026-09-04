"""Tests for fetched-pair outer crossmatches."""

import numpy as np
import pandas as pd
import pytest

import lsdb
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm


@pytest.mark.parametrize("suffix_method", ["all_columns", "overlapping_columns"])
def test_outer_recovers_fetched_right_rows(suffix_method, helpers):
    """Outer joins add unmatched primary-right rows from visited pixel pairs."""
    left_df = pd.DataFrame({"id": [1, 2], "ra": [0.0, 1.0], "dec": [0.0, 0.0]})
    right_df = pd.DataFrame({"id": [10, 11], "ra": [0.0001, 2.0], "dec": [0.0, 0.0]})
    left = lsdb.from_dataframe(left_df, lowest_order=0, highest_order=0, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=0, highest_order=0, margin_threshold=30)

    outer = left.crossmatch(
        right,
        how="outer",
        radius_arcsec=1,
        suffixes=("_left", "_right"),
        suffix_method=suffix_method,
    )
    result = outer.compute()

    helpers.assert_schema_correct(outer)
    assert len(result) == 3
    assert set(result["id_left"].dropna()) == {1, 2}
    assert set(result["id_right"].dropna()) == {10, 11}
    assert len(result[result["id_left"].notna() & result["id_right"].notna()]) == 1

    right_only = result[result["id_left"].isna()]
    assert right_only["id_right"].tolist() == [11]
    assert right_only["ra_left"].tolist() == right_only["ra_right"].tolist()
    assert right_only["dec_left"].tolist() == right_only["dec_right"].tolist()
    assert right_only.index.tolist() == right.compute().query("id == 11").index.tolist()
    assert result.loc[result["id_right"].isna(), "_dist_arcsec"].isna().all()
    assert right_only["_dist_arcsec"].isna().all()


def test_outer_filters_coarse_right_partition_to_aligned_pixels():
    """A coarse right partition is not emitted once per finer left pixel."""
    ras = [0.0, 5.0, 10.0, 15.0]
    left_df = pd.DataFrame({"id": range(4), "ra": ras, "dec": [0.0] * 4})
    right_df = pd.DataFrame({"id": range(10, 14), "ra": [ra + 0.01 for ra in ras], "dec": [0.0] * 4})
    left = lsdb.from_dataframe(left_df, lowest_order=3, highest_order=3, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=0, highest_order=0, margin_threshold=30)

    result = left.crossmatch(
        right,
        how="outer",
        radius_arcsec=1,
        suffixes=("_left", "_right"),
        suffix_method="all_columns",
    ).compute()

    assert len(result) == 8
    assert set(result["id_left"].dropna()) == set(range(4))
    assert set(result["id_right"].dropna()) == set(range(10, 14))


def test_outer_does_not_scan_right_only_sky():
    """Fetched-pair outer recovery keeps left alignment instead of reading disjoint right pixels."""
    left_df = pd.DataFrame({"id": [1], "ra": [0.0], "dec": [0.0]})
    right_df = pd.DataFrame({"id": [10], "ra": [180.0], "dec": [0.0]})
    left = lsdb.from_dataframe(left_df, lowest_order=3, highest_order=3, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=3, highest_order=3, margin_threshold=30)

    outer = left.crossmatch(
        right,
        how="outer",
        radius_arcsec=1,
        suffixes=("_left", "_right"),
        suffix_method="all_columns",
    )
    result = outer.compute()

    assert result["id_left"].tolist() == [1]
    assert result["id_right"].isna().all()
    assert outer.get_healpix_pixels() == left.get_healpix_pixels()


def test_crossmatch_rejects_unknown_join_method(small_sky_catalog, small_sky_xmatch_catalog):
    with pytest.raises(ValueError, match="`how` needs"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, how="right")  # type: ignore[arg-type]


def test_outer_supports_algorithms_without_extra_columns():
    class NoExtraColumnsCrossmatch(AbstractCrossmatchAlgorithm):
        def perform_crossmatch(self, crossmatch_args):
            del crossmatch_args
            return np.array([0]), np.array([0]), pd.DataFrame(index=[0])

    left_df = pd.DataFrame({"id": [1, 2], "ra": [0.0, 1.0], "dec": [0.0, 0.0]})
    right_df = pd.DataFrame({"id": [10, 11], "ra": [0.0, 2.0], "dec": [0.0, 0.0]})
    left = lsdb.from_dataframe(left_df, lowest_order=0, highest_order=0, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=0, highest_order=0, margin_threshold=30)

    result = left.crossmatch(
        right,
        how="outer",
        algorithm=NoExtraColumnsCrossmatch(),
        suffix_method="all_columns",
    ).compute()

    assert len(result) == 3
