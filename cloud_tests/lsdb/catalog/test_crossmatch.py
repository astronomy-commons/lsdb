import pandas as pd
import pytest

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm


def test_kdtree_crossmatch(small_sky_catalog_cloud, small_sky_xmatch_catalog_cloud, xmatch_correct_cloud):
    xmatched = small_sky_catalog_cloud.crossmatch(small_sky_xmatch_catalog_cloud).compute()
    assert len(xmatched) == len(xmatch_correct_cloud)
    for _, correct_row in xmatch_correct_cloud.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_kdtree_crossmatch_thresh(small_sky_catalog_cloud, small_sky_xmatch_catalog_cloud, xmatch_correct_005_cloud):
    xmatched = small_sky_catalog_cloud.crossmatch(small_sky_xmatch_catalog_cloud, d_thresh=0.005).compute()
    assert len(xmatched) == len(xmatch_correct_005_cloud)
    for _, correct_row in xmatch_correct_005_cloud.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_kdtree_crossmatch_multiple_neighbors(
    small_sky_catalog_cloud, small_sky_xmatch_catalog_cloud, xmatch_correct_3n_2t_no_margin_cloud
):
    xmatched = small_sky_catalog_cloud.crossmatch(small_sky_xmatch_catalog_cloud, n_neighbors=3, d_thresh=2).compute()
    assert len(xmatched) == len(xmatch_correct_3n_2t_no_margin_cloud)
    for _, correct_row in xmatch_correct_3n_2t_no_margin_cloud.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[
            (xmatched["id_small_sky"] == correct_row["ss_id"])
            & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])
        ]
        assert len(xmatch_row) == 1
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_custom_crossmatch_algorithm(small_sky_catalog_cloud, small_sky_xmatch_catalog_cloud, xmatch_mock_cloud):
    xmatched = small_sky_catalog_cloud.crossmatch(
        small_sky_xmatch_catalog_cloud, algorithm=MockCrossmatchAlgorithm, mock_results=xmatch_mock_cloud
    ).compute()
    assert len(xmatched) == len(xmatch_mock_cloud)
    for _, correct_row in xmatch_mock_cloud.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_wrong_suffixes(small_sky_catalog_cloud, small_sky_xmatch_catalog_cloud):
    with pytest.raises(ValueError):
        small_sky_catalog_cloud.crossmatch(small_sky_xmatch_catalog_cloud, suffixes=("wrong",))


class MockCrossmatchAlgorithm(AbstractCrossmatchAlgorithm):
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
        out.set_index("_hipscat_index", inplace=True)
        out["_DIST"] = mock_results["dist"].to_numpy()

        return out
