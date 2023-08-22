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
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, dthresh=0.005).compute()
    assert len(xmatched) == len(xmatch_correct_005)
    for _, correct_row in xmatch_correct_005.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].values == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])


def test_kdtree_crossmatch_multiple_neighbors(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t):
    xmatched = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, n_neighbors=3, dthresh=2).compute()
    assert len(xmatched) == len(xmatch_correct_3n_2t)
    for _, correct_row in xmatch_correct_3n_2t.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].values
        xmatch_row = xmatched[(xmatched["id_small_sky"] == correct_row["ss_id"]) & (xmatched["id_small_sky_xmatch"] == correct_row["xmatch_id"])]
        assert len(xmatch_row) == 1
        assert xmatch_row["_DIST"].values == pytest.approx(correct_row["dist"])
