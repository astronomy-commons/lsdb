import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb
from lsdb import Catalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.bounded_kdtree_match import BoundedKdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.dask.merge_catalog_functions import align_catalogs


@pytest.mark.parametrize("algo", [KdTreeCrossmatch])
class TestCrossmatch:
    @staticmethod
    def test_kdtree_crossmatch(algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched_cat = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog, algorithm=algo, radius_arcsec=0.01 * 3600
            )
            assert isinstance(xmatched_cat._ddf, nd.NestedFrame)
            xmatched = xmatched_cat.compute()
        alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
        assert xmatched_cat.hc_structure.moc == alignment.moc
        assert xmatched_cat.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

        assert isinstance(xmatched, npd.NestedFrame)
        assert len(xmatched) == len(xmatch_correct)
        for _, correct_row in xmatch_correct.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_thresh(algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005):
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_005)
        for _, correct_row in xmatch_correct_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_multiple_neighbors(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t_no_margin
    ):
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                n_neighbors=3,
                radius_arcsec=2 * 3600,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t_no_margin)
        for _, correct_row in xmatch_correct_3n_2t_no_margin.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_multiple_neighbors_margin(
        algo, small_sky_catalog, small_sky_xmatch_dir, small_sky_xmatch_margin_dir, xmatch_correct_3n_2t
    ):
        small_sky_xmatch_catalog = lsdb.read_hats(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
        )
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600, algorithm=algo
        ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t)
        for _, correct_row in xmatch_correct_3n_2t.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_crossmatch_negative_margin(
        algo,
        small_sky_left_xmatch_catalog,
        small_sky_xmatch_dir,
        small_sky_xmatch_margin_dir,
        xmatch_correct_3n_2t_negative,
    ):
        small_sky_xmatch_catalog = lsdb.read_hats(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
        )
        xmatched = small_sky_left_xmatch_catalog.crossmatch(
            small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600, algorithm=algo
        ).compute()
        assert len(xmatched) == len(xmatch_correct_3n_2t_negative)
        for _, correct_row in xmatch_correct_3n_2t_negative.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky_left_xmatch"].to_numpy()
            xmatch_row = xmatched[
                (xmatched["id_small_sky_left_xmatch"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_wrong_suffixes(algo, small_sky_catalog, small_sky_xmatch_catalog):
        with pytest.raises(ValueError):
            small_sky_catalog.crossmatch(small_sky_xmatch_catalog, suffixes=("wrong",), algorithm=algo)

    @staticmethod
    def test_right_margin_missing(algo, small_sky_catalog, small_sky_xmatch_catalog):
        small_sky_xmatch_catalog.margin = None
        with pytest.raises(ValueError, match="Right catalog margin"):
            small_sky_catalog.crossmatch(small_sky_xmatch_catalog, algorithm=algo, require_right_margin=True)


@pytest.mark.parametrize("algo", [BoundedKdTreeCrossmatch])
class TestBoundedCrossmatch:
    @staticmethod
    def test_kdtree_crossmatch_min_thresh(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_002_005
    ):
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                min_radius_arcsec=0.002 * 3600,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_002_005)
        for _, correct_row in xmatch_correct_002_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_min_thresh_multiple_neighbors_margin(
        algo,
        small_sky_catalog,
        small_sky_xmatch_dir,
        small_sky_xmatch_margin_dir,
        xmatch_correct_05_2_3n_margin,
    ):
        small_sky_xmatch_catalog = lsdb.read_hats(
            small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
        )
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            n_neighbors=3,
            radius_arcsec=2 * 3600,
            min_radius_arcsec=0.5 * 3600,
            algorithm=algo,
        ).compute()
        assert len(xmatched) == len(xmatch_correct_05_2_3n_margin)
        for _, correct_row in xmatch_correct_05_2_3n_margin.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[
                (xmatched["id_small_sky"] == correct_row["ss_id"])
                & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
            ]
            assert len(xmatch_row) == 1
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_kdtree_crossmatch_no_close_neighbors(
        algo, small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005
    ):
        # Set a very small minimum radius so that there is not a single point
        # with a very close neighbor
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                radius_arcsec=0.005 * 3600,
                min_radius_arcsec=1,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == len(xmatch_correct_005)
        for _, correct_row in xmatch_correct_005.iterrows():
            assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
            xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
            assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
            assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    @staticmethod
    def test_crossmatch_more_neighbors_than_points_available(
        algo, small_sky_catalog, small_sky_xmatch_catalog
    ):
        # The small_sky_xmatch catalog has 3 partitions (2 of length 41 and 1 of length 29).
        # Let's use n_neighbors above that to request more neighbors than there are points available.
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_xmatch_catalog,
                n_neighbors=50,
                radius_arcsec=2 * 3600,
                min_radius_arcsec=0.5 * 3600,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == 72
        assert all(xmatched.groupby("id_small_sky").size()) <= 50

    @staticmethod
    def test_self_crossmatch(algo, small_sky_catalog, small_sky_dir):
        # Read a second small sky catalog to not have duplicate labels
        small_sky_catalog_2 = lsdb.read_hats(small_sky_dir)
        small_sky_catalog_2.hc_structure.catalog_name = "small_sky_2"
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_catalog.crossmatch(
                small_sky_catalog_2,
                min_radius_arcsec=0,
                radius_arcsec=0.005 * 3600,
                algorithm=algo,
            ).compute()
        assert len(xmatched) == len(small_sky_catalog.compute())
        assert all(xmatched["_dist_arcsec"] == 0)

    @staticmethod
    def test_crossmatch_empty_left_partition(algo, small_sky_order1_catalog, small_sky_xmatch_catalog):
        ra = 300
        dec = -60
        radius_arcsec = 3 * 3600
        cone = small_sky_order1_catalog.cone_search(ra, dec, radius_arcsec)
        assert len(cone.get_healpix_pixels()) == 2
        assert len(cone.get_partition(1, 44)) == 5
        # There is an empty partition in the left catalog
        assert len(cone.get_partition(1, 46)) == 0
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = cone.crossmatch(
                small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600, algorithm=algo
            ).compute()
        assert len(xmatched) == 3
        assert all(xmatched["_dist_arcsec"] <= 0.01 * 3600)

    @staticmethod
    def test_crossmatch_empty_right_partition(algo, small_sky_order1_catalog, small_sky_xmatch_catalog):
        ra = 300
        dec = -60
        radius_arcsec = 3.4 * 3600
        cone = small_sky_xmatch_catalog.cone_search(ra, dec, radius_arcsec)
        assert len(cone.get_healpix_pixels()) == 2
        assert len(cone.get_partition(1, 44)) == 5
        # There is an empty partition in the right catalog
        assert len(cone.get_partition(1, 46)) == 0
        with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
            xmatched = small_sky_order1_catalog.crossmatch(
                cone, radius_arcsec=0.01 * 3600, algorithm=algo
            ).compute()
        assert len(xmatched) == 3
        assert all(xmatched["_dist_arcsec"] <= 0.01 * 3600)


def test_crossmatch_with_moc(small_sky_order1_catalog):
    order = 1
    pixels = [44, 45, 46]
    partitions = [small_sky_order1_catalog.get_partition(order, p).compute() for p in pixels]
    df = pd.concat(partitions)
    subset_catalog = lsdb.from_dataframe(df, lowest_order=0, highest_order=5)
    assert subset_catalog.get_healpix_pixels() == [HealpixPixel(0, 11)]
    xmatched = small_sky_order1_catalog.crossmatch(subset_catalog)
    assert xmatched.get_healpix_pixels() == [HealpixPixel(order, p) for p in pixels]
    xmatched = subset_catalog.crossmatch(small_sky_order1_catalog)
    assert xmatched.get_healpix_pixels() == [HealpixPixel(order, p) for p in pixels]


# pylint: disable=too-few-public-methods, unused-argument
class MockCrossmatchAlgorithm(AbstractCrossmatchAlgorithm):
    """Mock class used to test a crossmatch algorithm"""

    extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=np.float64)})

    def perform_crossmatch(self, mock_results: pd.DataFrame = None):
        left_reset = self.left.reset_index(drop=True)
        right_reset = self.right.reset_index(drop=True)
        mock_results = mock_results[mock_results["ss_id"].isin(left_reset["id"].to_numpy())]
        left_indexes = mock_results.apply(
            lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1
        )
        right_indexes = mock_results.apply(
            lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1
        )
        extra_columns = pd.DataFrame({"_DIST": mock_results["dist"]})

        return left_indexes.to_numpy(), right_indexes.to_numpy(), extra_columns

    @classmethod
    def validate(cls, left: Catalog, right: Catalog, mock_results: pd.DataFrame = None):
        super().validate(left, right)


def test_custom_crossmatch_algorithm(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithm, mock_results=xmatch_mock
        ).compute()
    assert len(xmatched) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].to_numpy() == pytest.approx(correct_row["dist"])


# pylint: disable=too-few-public-methods, arguments-differ, unused-argument
class MockCrossmatchAlgorithmOverwrite(AbstractCrossmatchAlgorithm):
    """Mock class used to test a crossmatch algorithm"""

    extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=np.float64)})

    def crossmatch(self, mock_results: pd.DataFrame = None):  # type: ignore
        left_reset = self.left.reset_index(drop=True)
        right_reset = self.right.reset_index(drop=True)
        self._rename_columns_with_suffix(self.left, self.suffixes[0])
        self._rename_columns_with_suffix(self.right, self.suffixes[1])
        mock_results = mock_results[mock_results["ss_id"].isin(left_reset["id"].to_numpy())]
        left_indexes = mock_results.apply(
            lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1
        )
        right_indexes = mock_results.apply(
            lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1
        )
        left_join_part = self.left.iloc[left_indexes.to_numpy()].reset_index()
        right_join_part = self.right.iloc[right_indexes.to_numpy()].reset_index(drop=True)
        out = pd.concat(
            [
                left_join_part,  # select the rows of the left table
                right_join_part,  # select the rows of the right table
            ],
            axis=1,
        )
        out.set_index(SPATIAL_INDEX_COLUMN, inplace=True)
        extra_columns = pd.DataFrame({"_DIST": mock_results["dist"]})
        self._append_extra_columns(out, extra_columns)
        return out

    @classmethod
    def validate(cls, left: Catalog, right: Catalog, mock_results: pd.DataFrame = None):
        super().validate(left, right)


def test_custom_crossmatch_algorithm_overwrite(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithmOverwrite, mock_results=xmatch_mock
        ).compute()
    assert len(xmatched) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].to_numpy() == pytest.approx(correct_row["dist"])


def test_append_extra_columns_have_correct_type(small_sky_xmatch_catalog):
    algo = MockCrossmatchAlgorithm
    # Set a pyarrow type for the extra column
    pa_float64_dtype = pd.ArrowDtype(pa.float64())
    algo.extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=pa_float64_dtype)})
    # Create mock values for extra_columns
    xmatch_df = small_sky_xmatch_catalog.compute()
    dist_values = np.arange(len(xmatch_df))
    # Check that the extra column keeps its type
    extra_columns = {"_DIST": pd.Series(dist_values, dtype=pa_float64_dtype)}
    algo._append_extra_columns(xmatch_df, pd.DataFrame(extra_columns))
    assert "_DIST" in xmatch_df and xmatch_df["_DIST"].dtype == pa_float64_dtype


def test_append_extra_columns_raises_value_error(small_sky_xmatch_catalog):
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


def test_algorithm_has_no_extra_columns_specified(small_sky_xmatch_catalog):
    algo = MockCrossmatchAlgorithm
    # Create mock values for extra_columns
    xmatch_df = small_sky_xmatch_catalog.compute()
    dist_values = np.arange(len(xmatch_df))
    extra_columns = {"_DIST": pd.Series(dist_values, dtype=np.dtype("float64"))}
    # The crossmatch algorithm has no extra_columns specified
    algo.extra_columns = None
    algo._append_extra_columns(xmatch_df, pd.DataFrame(extra_columns))
    assert "_DIST" not in xmatch_df.columns


def test_raise_for_unknown_kwargs(small_sky_catalog):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        small_sky_catalog.crossmatch(small_sky_catalog, unknown_kwarg="value")
