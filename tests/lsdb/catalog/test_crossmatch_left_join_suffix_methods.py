"""Tests for left-join crossmatch with different suffix methods when right catalog is empty/partial."""

import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties

from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


class TestLeftJoinSuffixMethods:
    """Test left-join behavior with different suffix methods when right catalog doesn't match everything."""

    @staticmethod
    def create_test_frames():
        """Create simple test frames for crossmatch testing."""
        # Left catalog with 5 points
        left_data = {
            "ra": [0.0, 1.0, 2.0, 3.0, 4.0],
            "dec": [0.0, 1.0, 2.0, 3.0, 4.0],
            "id": [100, 101, 102, 103, 104],
            "common_col": ["a", "b", "c", "d", "e"],
            "left_only": [10, 20, 30, 40, 50],
        }
        left = npd.NestedFrame(pd.DataFrame(left_data))

        # Right catalog with 2 points that match first two left points
        right_data = {
            "ra": [0.001, 1.001],  # Close matches to first two left points
            "dec": [0.001, 1.001],
            "id": [200, 201],
            "common_col": ["x", "y"],
            "right_only": [100, 200],
        }
        right = npd.NestedFrame(pd.DataFrame(right_data))

        # Mock catalog info
        left_info = TableProperties(
            catalog_name="left_cat",
            catalog_type="object",
            total_rows=5,
            ra_column="ra",
            dec_column="dec",
        )
        right_info = TableProperties(
            catalog_name="right_cat",
            catalog_type="object",
            total_rows=2,
            ra_column="ra",
            dec_column="dec",
        )

        return left, right, left_info, right_info

    def test_left_join_all_columns_suffix_no_right_matches(self):
        """Test left-join with all_columns suffix when right catalog is empty (no matches)."""
        left, _, left_info, right_info = self.create_test_frames()

        # Create empty right (simulates partition with no right data)
        empty_right = npd.NestedFrame(
            pd.DataFrame(
                {
                    "ra": pd.Series(dtype=float),
                    "dec": pd.Series(dtype=float),
                    "id": pd.Series(dtype="int64"),
                    "common_col": pd.Series(dtype=object),
                    "right_only": pd.Series(dtype="int64"),
                }
            )
        )

        suffixes = ("_left", "_right")
        crossmatch_args = CrossmatchArgs(
            left_df=left,
            right_df=empty_right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)

        # Perform crossmatch with all_columns suffix
        result = algo.crossmatch(crossmatch_args, how="left", suffixes=suffixes, suffix_method="all_columns")

        # Assertions
        assert len(result) == 5, "Should have all 5 left rows"
        assert "ra_left" in result.columns
        assert "dec_left" in result.columns
        assert "id_left" in result.columns
        assert "common_col_left" in result.columns
        assert "left_only_left" in result.columns

        # Right columns should exist with suffix and be all NaN
        assert "ra_right" in result.columns
        assert "dec_right" in result.columns
        assert "id_right" in result.columns
        assert "common_col_right" in result.columns
        assert "right_only_right" in result.columns

        assert result["ra_right"].isna().all(), "All right ra values should be NaN"
        assert result["id_right"].isna().all(), "All right id values should be NaN"
        assert result["right_only_right"].isna().all(), "All right_only values should be NaN"

        # Left data should be intact
        assert result["id_left"].tolist() == [100, 101, 102, 103, 104]

    def test_left_join_overlapping_columns_suffix_no_right_matches(self):
        """Test left-join with overlapping_columns suffix when right catalog is empty."""
        left, _, left_info, right_info = self.create_test_frames()

        # Create empty right
        empty_right = npd.NestedFrame(
            pd.DataFrame(
                {
                    "ra": pd.Series(dtype=float),
                    "dec": pd.Series(dtype=float),
                    "id": pd.Series(dtype="int64"),
                    "common_col": pd.Series(dtype=object),
                    "right_only": pd.Series(dtype="int64"),
                }
            )
        )

        suffixes = ("_left", "_right")
        crossmatch_args = CrossmatchArgs(
            left_df=left,
            right_df=empty_right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)

        # Perform crossmatch with overlapping_columns suffix
        result = algo.crossmatch(
            crossmatch_args, how="left", suffixes=suffixes, suffix_method="overlapping_columns"
        )

        # Assertions
        assert len(result) == 5, "Should have all 5 left rows"

        # With overlapping_columns:
        # - ra, dec, id, common_col overlap => get suffixed
        # - left_only is unique to left => keeps original name
        # - right_only is unique to right => keeps original name
        assert "ra_left" in result.columns
        assert "dec_left" in result.columns
        assert "id_left" in result.columns
        assert "common_col_left" in result.columns
        assert "left_only" in result.columns  # No suffix (unique to left)

        assert "ra_right" in result.columns
        assert "dec_right" in result.columns
        assert "id_right" in result.columns
        assert "common_col_right" in result.columns
        assert "right_only" in result.columns  # No suffix (unique to right)

        # Right columns should be all NaN
        assert result["ra_right"].isna().all()
        assert result["id_right"].isna().all()
        assert result["right_only"].isna().all()

        # Left data should be intact
        assert result["id_left"].tolist() == [100, 101, 102, 103, 104]
        assert result["left_only"].tolist() == [10, 20, 30, 40, 50]

    def test_left_join_all_columns_suffix_partial_matches(self):
        """Test left-join with all_columns suffix when only some left rows match."""
        left, right, left_info, right_info = self.create_test_frames()

        suffixes = ("_left", "_right")
        crossmatch_args = CrossmatchArgs(
            left_df=left,
            right_df=right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)

        # Large radius so first two points match
        result = algo.crossmatch(crossmatch_args, how="left", suffixes=suffixes, suffix_method="all_columns")

        # Should have 5 or more rows (2 matches + 3 unmatched, or more if multiple matches per left)
        assert len(result) >= 5, f"Should have at least 5 rows, got {len(result)}"

        # Check that we have both matched and unmatched rows
        matched_mask = result["id_right"].notna()
        n_matched = matched_mask.sum()
        n_unmatched = (~matched_mask).sum()

        assert n_matched >= 2, f"Should have at least 2 matched rows, got {n_matched}"
        assert n_unmatched >= 3, f"Should have at least 3 unmatched rows, got {n_unmatched}"

        # Check column names
        assert "ra_left" in result.columns
        assert "id_left" in result.columns
        assert "common_col_left" in result.columns
        assert "left_only_left" in result.columns
        assert "ra_right" in result.columns
        assert "id_right" in result.columns
        assert "common_col_right" in result.columns
        assert "right_only_right" in result.columns

        # Unmatched rows should have NaN in right columns
        unmatched = result[~matched_mask]
        assert unmatched["id_right"].isna().all()
        assert unmatched["right_only_right"].isna().all()

        # Matched rows should have non-NaN right values
        matched = result[matched_mask]
        assert matched["id_right"].notna().all()

        # All left IDs should be present
        unique_left_ids = result["id_left"].unique()
        assert set(unique_left_ids) == {100, 101, 102, 103, 104}

    def test_left_join_overlapping_columns_suffix_partial_matches(self):
        """Test left-join with overlapping_columns suffix when only some left rows match."""
        left, right, left_info, right_info = self.create_test_frames()

        suffixes = ("_left", "_right")
        crossmatch_args = CrossmatchArgs(
            left_df=left,
            right_df=right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)

        # Large radius so first two points match
        result = algo.crossmatch(
            crossmatch_args, how="left", suffixes=suffixes, suffix_method="overlapping_columns"
        )

        # Should have 5 or more rows
        assert len(result) >= 5

        # Check that we have both matched and unmatched rows
        matched_mask = result["id_right"].notna()
        n_matched = matched_mask.sum()
        n_unmatched = (~matched_mask).sum()

        assert n_matched >= 2, f"Should have at least 2 matched rows, got {n_matched}"
        assert n_unmatched >= 3, f"Should have at least 3 unmatched rows, got {n_unmatched}"

        # Check column names (overlapping columns get suffixed, unique ones don't)
        assert "ra_left" in result.columns
        assert "id_left" in result.columns
        assert "common_col_left" in result.columns
        assert "left_only" in result.columns  # No suffix
        assert "ra_right" in result.columns
        assert "id_right" in result.columns
        assert "common_col_right" in result.columns
        assert "right_only" in result.columns  # No suffix

        # Unmatched rows should have NaN in right columns
        unmatched = result[~matched_mask]
        assert unmatched["id_right"].isna().all()
        assert unmatched["right_only"].isna().all()

        # Matched rows should have non-NaN right values
        matched = result[matched_mask]
        assert matched["id_right"].notna().all()

        # All left IDs should be present
        unique_left_ids = result["id_left"].unique()
        assert set(unique_left_ids) == {100, 101, 102, 103, 104}

    def test_extra_columns_alignment_with_suffix_methods(self):
        """Test that extra columns (e.g., _dist_arcsec) align correctly with different suffix methods."""
        left, right, left_info, right_info = self.create_test_frames()

        suffixes = ("_left", "_right")

        # Test with all_columns
        crossmatch_args_all = CrossmatchArgs(
            left_df=left,
            right_df=right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo_all = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)
        result_all = algo_all.crossmatch(
            crossmatch_args_all, how="left", suffixes=suffixes, suffix_method="all_columns"
        )

        # _dist_arcsec should exist
        assert "_dist_arcsec" in result_all.columns

        # Matched rows should have non-NaN distance
        matched_mask = result_all["id_right"].notna()
        matched_dist = result_all.loc[matched_mask, "_dist_arcsec"]
        assert matched_dist.notna().all(), "Matched rows should have distance values"

        # Unmatched rows should have NaN distance
        unmatched_dist = result_all.loc[~matched_mask, "_dist_arcsec"]
        assert unmatched_dist.isna().all(), "Unmatched rows should have NaN distance"

        # Test with overlapping_columns
        crossmatch_args_overlap = CrossmatchArgs(
            left_df=left,
            right_df=right,
            left_order=0,
            left_pixel=0,
            right_order=0,
            right_pixel=0,
            left_catalog_info=left_info,
            right_catalog_info=right_info,
            right_margin_catalog_info=None,
        )
        algo_overlap = KdTreeCrossmatch(n_neighbors=1, radius_arcsec=10)
        result_overlap = algo_overlap.crossmatch(
            crossmatch_args_overlap, how="left", suffixes=suffixes, suffix_method="overlapping_columns"
        )

        # _dist_arcsec should exist
        assert "_dist_arcsec" in result_overlap.columns

        # Same alignment checks
        matched_mask_overlap = result_overlap["id_right"].notna()
        matched_dist_overlap = result_overlap.loc[matched_mask_overlap, "_dist_arcsec"]
        assert matched_dist_overlap.notna().all()

        unmatched_dist_overlap = result_overlap.loc[~matched_mask_overlap, "_dist_arcsec"]
        assert unmatched_dist_overlap.isna().all()
