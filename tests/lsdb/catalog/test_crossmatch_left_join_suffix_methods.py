"""Tests for left-join crossmatch with different suffix methods when right catalog is empty/partial."""

import pandas as pd
import pytest

import lsdb


class TestLeftJoinSuffixMethods:
    """Test left-join behavior with different suffix methods when right catalog doesn't match everything."""

    @staticmethod
    def create_test_catalogs():
        """Create simple test catalogs for crossmatch testing."""
        # Left catalog with 5 points
        left_data = {
            "ra": [0.0, 1.0, 2.0, 3.0, 4.0],
            "dec": [0.0, 1.0, 2.0, 3.0, 4.0],
            "id": [100, 101, 102, 103, 104],
            "common_col": ["a", "b", "c", "d", "e"],
            "left_only": [10, 20, 30, 40, 50],
        }
        left_df = pd.DataFrame(left_data)

        # Right catalog with 2 points that match first two left points
        right_data = {
            "ra": [0.001, 1.001],  # Close matches to first two left points
            "dec": [0.001, 1.001],
            "id": [200, 201],
            "common_col": ["x", "y"],
            "right_only": [100, 200],
        }
        right_df = pd.DataFrame(right_data)

        # Create catalogs using from_dataframe with margin_threshold >= radius_arcsec
        left_catalog = lsdb.from_dataframe(
            left_df,
            ra_column="ra",
            dec_column="dec",
            lowest_order=0,
            highest_order=0,
            margin_threshold=30,
        )
        right_catalog = lsdb.from_dataframe(
            right_df,
            ra_column="ra",
            dec_column="dec",
            lowest_order=0,
            highest_order=0,
            margin_threshold=30,
        )

        return left_catalog, right_catalog

    @staticmethod
    def create_distant_right_catalog():
        """Create a right catalog that is far away from the left catalog (no matches)."""
        # Use coordinates far from left catalog (left is at ~0-4 degrees)
        far_right_df = pd.DataFrame(
            {
                "ra": [180.0, 181.0],  # Far away from left catalog
                "dec": [80.0, 81.0],
                "id": [200, 201],
                "common_col": ["x", "y"],
                "right_only": [100, 200],
            }
        )
        return lsdb.from_dataframe(
            far_right_df,
            ra_column="ra",
            dec_column="dec",
            lowest_order=0,
            highest_order=0,
            margin_threshold=30,
        )

    @pytest.mark.parametrize(
        "suffix_method",
        ["all_columns", "overlapping_columns"],
    )
    def test_left_join_no_right_matches(self, suffix_method):
        """Test left-join when right catalog is far away (no matches), with different suffix methods."""
        left_catalog, _ = self.create_test_catalogs()
        far_right_catalog = self.create_distant_right_catalog()

        suffixes = ("_left", "_right")

        xmatched = left_catalog.crossmatch(
            far_right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method=suffix_method,
        )
        result = xmatched.compute()

        # Check unique column naming based on suffix method
        if suffix_method == "overlapping_columns":
            # overlapping_columns: unique columns get no suffix
            assert "left_only" in result.columns
            assert "right_only" in result.columns
        else:
            # all_columns: all columns get suffixed
            assert "left_only_left" in result.columns
            assert "right_only_right" in result.columns

        # All right columns should be NA
        assert result["ra_right"].isna().all()
        assert result["id_right"].isna().all()

        # Left data should be intact
        assert result["id_left"].tolist() == [100, 101, 102, 103, 104]

    @pytest.mark.parametrize(
        "suffix_method",
        ["all_columns", "overlapping_columns"],
    )
    def test_left_join_partial_matches(self, suffix_method):
        """Test left-join when only some left rows match, with different suffix methods."""
        left_catalog, right_catalog = self.create_test_catalogs()

        suffixes = ("_left", "_right")

        xmatched = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method=suffix_method,
        )
        result = xmatched.compute()

        # Should have 5 or more rows (2 matches + 3 unmatched, or more if multiple matches per left)
        assert len(result) >= 5, f"Should have at least 5 rows, got {len(result)}"

        # Check that we have both matched and unmatched rows
        matched_mask = result["id_right"].notna()
        n_matched = matched_mask.sum()
        n_unmatched = (~matched_mask).sum()

        assert n_matched >= 2, f"Should have at least 2 matched rows, got {n_matched}"
        assert n_unmatched >= 3, f"Should have at least 3 unmatched rows, got {n_unmatched}"

        # Check column names (all methods have these)
        assert "ra_left" in result.columns
        assert "id_left" in result.columns
        assert "common_col_left" in result.columns
        assert "ra_right" in result.columns
        assert "id_right" in result.columns
        assert "common_col_right" in result.columns

        # Check unique column naming based on suffix method
        if suffix_method == "overlapping_columns":
            # overlapping_columns: unique columns get no suffix
            assert "left_only" in result.columns
            assert "right_only" in result.columns
        else:
            # all_columns: all columns get suffixed
            assert "left_only_left" in result.columns
            assert "right_only_right" in result.columns

        # Unmatched rows should have NA in right columns
        unmatched = result[~matched_mask]
        assert unmatched["id_right"].isna().all()

        # Matched rows should have non-NA right values
        matched = result[matched_mask]
        assert matched["id_right"].notna().all()

        # All left IDs should be present
        unique_left_ids = result["id_left"].unique()
        assert set(unique_left_ids) == {100, 101, 102, 103, 104}

    @pytest.mark.parametrize(
        "suffix_method",
        ["all_columns", "overlapping_columns"],
    )
    def test_extra_columns_alignment_with_suffix_methods(self, suffix_method):
        """Test that extra columns (e.g., _dist_arcsec) align correctly with different suffix methods."""
        left_catalog, right_catalog = self.create_test_catalogs()

        suffixes = ("_left", "_right")

        xmatched = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method=suffix_method,
        )
        result = xmatched.compute()

        # _dist_arcsec should exist
        assert "_dist_arcsec" in result.columns

        # Matched rows should have non-NA distance
        matched_mask = result["id_right"].notna()
        matched_dist = result.loc[matched_mask, "_dist_arcsec"]
        assert matched_dist.notna().all(), "Matched rows should have distance values"

        # Unmatched rows should have NA distance
        unmatched_dist = result.loc[~matched_mask, "_dist_arcsec"]
        assert unmatched_dist.isna().all(), "Unmatched rows should have NA distance"
