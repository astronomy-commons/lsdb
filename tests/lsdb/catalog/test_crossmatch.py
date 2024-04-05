import numpy as np
import pandas as pd
import pytest
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN

import lsdb
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.bounded_kdtree_match import BoundedKdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


@pytest.mark.parametrize("algo", [KdTreeCrossmatch])
class TestCrossmatch:
    @staticmethod
    def test_kdtree_crossmatch(algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                algorithm=algo,
                radius_arcsec=0.01 * 3600,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == len(xmatch_correct)
        for _, correct_row in xmatch_correct.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_thresh(algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005):
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                algorithm=algo,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_005)
        for _, correct_row in xmatch_correct_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_multiple_neighbors(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t_no_margin
    ):
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                n_neighbors=3,
                radius_arcsec=2 * 3600,
                algorithm=algo,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t_no_margin)
        for _, correct_row in xmatch_correct_3n_2t_no_margin.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_multiple_neighbors_margin(
        algo, small_sky_catalog, small_sky_xmatch_dir, small_sky_xmatch_margin_catalog, xmatch_correct_3n_2t
    ):
        small_sky_xmatch_catalog = lsdb.read_hipscat(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_catalog
        )
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600, algorithm=algo
        ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t)
        for _, correct_row in xmatch_correct_3n_2t.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_crossmatch_negative_margin(
        algo,
        small_sky_left_xmatch_catalog,
        small_sky_xmatch_dir,
        small_sky_xmatch_margin_catalog,
        xmatch_correct_3n_2t_negative,
    ):
        small_sky_xmatch_catalog = lsdb.read_hipscat(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_catalog
        )
        xmatched = small_sky_left_xmatch_catalog.crossmatch(
            small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600, algorithm=algo
        ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t_negative)
        for _, correct_row in xmatch_correct_3n_2t_negative.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky_left_xmatch"].values
            xmatch_row = xmatched[
                (xmatched["id_small_sky_left_xmatch"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_wrong_suffixes(algo, small_sky_catalog, small_sky_xmatch_catalog):
        with pytest.raises(ValueError):
            small_sky_catalog.crossmatch(small_sky_xmatch_catalog, suffixes=("wrong",), algorithm=algo)


@pytest.mark.parametrize("algo", [BoundedKdTreeCrossmatch])
class TestBoundedCrossmatch:
    @staticmethod
    def test_kdtree_crossmatch_min_thresh(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_002_005
    ):
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                min_radius_arcsec=0.002 * 3600,
                algorithm=algo,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_002_005)
        for _, correct_row in xmatch_correct_002_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_min_thresh_multiple_neighbors_margin(
        algo,
        small_sky_catalog,
        small_sky_xmatch_dir,
        small_sky_xmatch_margin_catalog,
        xmatch_correct_05_2_3n_margin,
    ):
        small_sky_xmatch_catalog = lsdb.read_hipscat(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_catalog
        )
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            n_neighbors=3,
            radius_arcsec=2 * 3600,
            min_radius_arcsec=0.5 * 3600,
            algorithm=algo,
            require_right_margin=False,
        ).compute()
        assert len(xmatched) == len(xmatch_correct_05_2_3n_margin)
        for _, correct_row in xmatch_correct_05_2_3n_margin.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_no_close_neighbors(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005
    ):
        # Set a very small minimum radius so that there is not a single point
        # with a very close neighbor
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                min_radius_arcsec=1,
                algorithm=algo,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_005)
        for _, correct_row in xmatch_correct_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].values
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].values == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_crossmatch_more_neighbors_than_points_available(
        algo, small_sky_catalog, small_sky_xmatch_catalog
    ):
        # The small_sky_xmatch catalog has 3 partitions (2 of length 41 and 1 of length 29).
        # Let's use n_neighbors above that to request more neighbors than there are points available.
        with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                n_neighbors=50,
                radius_arcsec=2 * 3600,
                min_radius_arcsec=0.5 * 3600,
                algorithm=algo,
                require_right_margin=False,
            ).compute()
        assert len(xmatched) == 72
        assert all(xmatched.groupby("id_small_sky").size()) <= 50

    @staticmethod
    def test_self_crossmatch(algo, small_sky_catalog, small_sky_dir):
        # Read a second small sky catalog to not have duplicate labels
        small_sky_catalog_2 = lsdb.read_hipscat(small_sky_dir)
        small_sky_catalog_2.hc_structure.catalog_name = "small_sky_2"
        xmatched = small_sky_catalog.crossmatch(
            small_sky_catalog_2,
            min_radius_arcsec=0,
            radius_arcsec=0.005 * 3600,
            algorithm=algo,
            require_right_margin=False,
        ).compute()
        assert len(xmatched) == len(small_sky_catalog.compute())
        assert all(xmatched["_dist_arcsec"] == 0)


# pylint: disable=too-few-public-methods
class MockCrossmatchAlgorithm(AbstractCrossmatchAlgorithm):
    """Mock class used to test a crossmatch algorithm"""

    extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=np.dtype("float64"))})

    def crossmatch(self, mock_results: pd.DataFrame = None):
        left_reset = self.left.reset_index(drop=True)
        right_reset = self.right.reset_index(drop=True)
        self._rename_columns_with_suffix(self.left, self.suffixes[0])
        self._rename_columns_with_suffix(self.right, self.suffixes[1])
        mock_results = mock_results[mock_results["ss_id"].isin(left_reset["id"].values)]
        left_indexes = mock_results.apply(
            lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1
        )
        right_indexes = mock_results.apply(
            lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1
        )
        left_join_part = self.left.iloc[left_indexes.values].reset_index()
        right_join_part = self.right.iloc[right_indexes.values].reset_index(drop=True)
        out = pd.concat(
            [
                left_join_part,  # select the rows of the left table
                right_join_part,  # select the rows of the right table
            ],
            axis=1,
        )
        out.set_index(HIPSCAT_ID_COLUMN, inplace=True)
        extra_columns = pd.DataFrame({"_DIST": mock_results["dist"]})
        self._append_extra_columns(out, extra_columns)

        return out


def test_custom_crossmatch_algorithm(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithm, mock_results=xmatch_mock
        ).compute()
    assert len(xmatched) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_append_extra_columns(small_sky_xmatch_catalog):
    algo = MockCrossmatchAlgorithm
    # Create mock values for extra_columns
    xmatch_df = small_sky_xmatch_catalog.compute()
    dist_values = np.arange(len(xmatch_df))
    extra_columns = {"_DIST": pd.Series(dist_values, dtype=np.dtype("float64"))}
    # At least a provided column is not in the specification
    with pytest.raises(ValueError, match="Provided extra column"):
        no_specified_columns = {"_DIST_2": extra_columns["_DIST"]}
        algo._append_extra_columns(xmatch_df, pd.DataFrame(no_specified_columns))
    # At least an extra column is missing
    with pytest.raises(ValueError, match="Missing extra column"):
        missing_columns = {"_DIST_2": extra_columns["_DIST"]}
        algo.extra_columns = pd.DataFrame({**extra_columns, **missing_columns})
        algo._append_extra_columns(xmatch_df, pd.DataFrame(extra_columns))
    # At least an extra column is of invalid type
    with pytest.raises(ValueError, match="Invalid type"):
        invalid_type_columns = {"_DIST": extra_columns["_DIST"].astype("int64")}
        algo._append_extra_columns(xmatch_df, pd.DataFrame(invalid_type_columns))
    # No extra_columns were specified
    with pytest.raises(ValueError, match="No extra column values"):
        algo._append_extra_columns(xmatch_df, extra_columns=None)
    # The crossmatch algorithm has no extra_columns specified
    algo.extra_columns = None
    algo._append_extra_columns(xmatch_df, pd.DataFrame(extra_columns))
    assert "_DIST" not in xmatch_df.columns
