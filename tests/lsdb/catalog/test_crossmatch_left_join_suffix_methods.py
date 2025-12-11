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

    def test_left_join_all_columns_suffix_no_right_matches(self):
        """Test left-join with all_columns suffix when right catalog is far away (no matches)."""
        left_catalog, _ = self.create_test_catalogs()
        far_right_catalog = self.create_distant_right_catalog()

        suffixes = ("_left", "_right")

        # Perform crossmatch with all_columns suffix using catalog API
        xmatched = left_catalog.crossmatch(
            far_right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="all_columns",
        )
        result = xmatched.compute()

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

    @pytest.mark.xfail(
        reason="Bug: overlapping_columns suffix method doesn't add unique right columns when no matches exist"
    )
    def test_left_join_overlapping_columns_suffix_no_right_matches(self):
        """Test left-join with overlapping_columns suffix when right catalog is far away (no matches)."""
        left_catalog, _ = self.create_test_catalogs()
        far_right_catalog = self.create_distant_right_catalog()

        suffixes = ("_left", "_right")

        # Perform crossmatch with overlapping_columns suffix using catalog API
        xmatched = left_catalog.crossmatch(
            far_right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="overlapping_columns",
        )
        result = xmatched.compute()

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
        left_catalog, right_catalog = self.create_test_catalogs()

        suffixes = ("_left", "_right")

        # Large radius so first two points match
        xmatched = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="all_columns",
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
        left_catalog, right_catalog = self.create_test_catalogs()

        suffixes = ("_left", "_right")

        # Large radius so first two points match
        xmatched = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="overlapping_columns",
        )
        result = xmatched.compute()

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
        left_catalog, right_catalog = self.create_test_catalogs()

        suffixes = ("_left", "_right")

        # Test with all_columns
        xmatched_all = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="all_columns",
        )
        result_all = xmatched_all.compute()

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
        xmatched_overlap = left_catalog.crossmatch(
            right_catalog,
            how="left",
            radius_arcsec=10,
            suffixes=suffixes,
            suffix_method="overlapping_columns",
        )
        result_overlap = xmatched_overlap.compute()

        # _dist_arcsec should exist
        assert "_dist_arcsec" in result_overlap.columns

        # Same alignment checks
        matched_mask_overlap = result_overlap["id_right"].notna()
        matched_dist_overlap = result_overlap.loc[matched_mask_overlap, "_dist_arcsec"]
        assert matched_dist_overlap.notna().all()

        unmatched_dist_overlap = result_overlap.loc[~matched_mask_overlap, "_dist_arcsec"]
        assert unmatched_dist_overlap.isna().all()
