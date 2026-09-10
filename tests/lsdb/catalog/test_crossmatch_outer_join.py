"""Tests for outer crossmatches.

An outer crossmatch aligns pixels in the outer way: every pixel of either catalog is
visited, right rows in sky without left coverage are emitted unmatched, and right rows
are emitted unmatched only when no left row pairs with them anywhere — including across
pixel boundaries, via the left catalog's margin cache and boundary partitions.
"""

import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.healpix_shim import radec2pix

import lsdb
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import (
    AbstractCrossmatchAlgorithm,
)
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch

ORDER = 2
RADIUS = 10


def _pix(order, ra, dec=0.0):
    return radec2pix(order, ra, dec)[0].item()


def _adjacent_ra_pair():
    """Return nearby coordinates on opposite sides of an order-2 pixel boundary."""
    assert _pix(ORDER, 0.0) != _pix(ORDER, 0.001)
    return 0.0, 0.001


def _catalog(df, name, margin_threshold=60, order=ORDER):
    return lsdb.from_dataframe(
        df,
        catalog_name=name,
        ra_column="ra",
        dec_column="dec",
        lowest_order=order,
        highest_order=order,
        margin_threshold=margin_threshold,
    )


@pytest.mark.parametrize("suffix_method", ["all_columns", "overlapping_columns"])
def test_outer_recovers_unmatched_right_rows(suffix_method, helpers):
    """Outer joins add unmatched primary-right rows alongside matched and left-only rows."""
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
    result = pd.DataFrame(outer.compute())

    helpers.assert_schema_correct(outer)
    assert len(result) == 3
    assert set(result["id_left"].dropna()) == {1, 2}
    assert set(result["id_right"].dropna()) == {10, 11}
    assert len(result[result["id_left"].notna() & result["id_right"].notna()]) == 1

    right_only = result[result["id_left"].isna()]
    assert right_only["id_right"].tolist() == [11]
    assert right_only["ra_left"].tolist() == right_only["ra_right"].tolist()
    assert right_only["dec_left"].tolist() == right_only["dec_right"].tolist()
    assert right_only.index.tolist() == pd.DataFrame(right.compute()).query("id == 11").index.tolist()
    unmatched = result["id_left"].isna() | result["id_right"].isna()
    assert result.loc[unmatched, "_dist_arcsec"].isna().all()


def test_outer_scans_right_only_sky():
    """Sky covered only by the right catalog is scanned and emitted unmatched."""
    left_df = pd.DataFrame({"id": [1], "ra": [0.0], "dec": [0.0]})
    right_df = pd.DataFrame({"id": [10], "ra": [120.0], "dec": [10.0]})
    left = lsdb.from_dataframe(left_df, lowest_order=3, highest_order=3, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=3, highest_order=3, margin_threshold=30)

    outer = left.crossmatch(
        right,
        how="outer",
        radius_arcsec=1,
        suffixes=("_left", "_right"),
        suffix_method="all_columns",
    )
    result = pd.DataFrame(outer.compute())

    assert len(result) == 2
    assert result["id_left"].dropna().tolist() == [1]
    assert result[result["id_left"].isna()]["id_right"].dropna().tolist() == [10]
    assert set(outer.get_healpix_pixels()) == set(left.get_healpix_pixels()) | set(right.get_healpix_pixels())


def test_outer_right_only_pixel_boundary_row_matched_once():
    """A right row just across left coverage is matched once, not also emitted unmatched.

    The right row lives in a right-only pixel but is within the radius of a left row in
    an adjacent pixel: the left partitions bordering the right-only pixel see it, so it
    must appear as a match exactly once, and the far right row must be emitted unmatched.
    """
    ra_left, ra_boundary = _adjacent_ra_pair()
    left = _catalog(pd.DataFrame({"id": [1], "ra": [ra_left], "dec": [0.0]}), "left")
    right = _catalog(pd.DataFrame({"id": [10, 11], "ra": [ra_boundary, 120.0], "dec": [0.0, 10.0]}), "right")

    result = pd.DataFrame(
        left.crossmatch(
            right,
            how="outer",
            radius_arcsec=RADIUS,
            suffixes=("_left", "_right"),
            suffix_method="all_columns",
        ).compute()
    )

    assert len(result) == 2
    matched = result[result["id_right"] == 10]
    assert len(matched) == 1
    assert matched["id_left"].tolist() == [1]
    assert matched["_dist_arcsec"].item() == pytest.approx(3.6, abs=0.1)
    right_only = result[result["id_left"].isna()]
    assert right_only["id_right"].tolist() == [11]


def test_outer_right_only_pixel_gathers_multiple_left_partitions():
    """A right-only pixel ring-touching two left partitions gathers and concatenates both.

    Its task reads several adjacent left partitions through one multi-pixel input slot,
    so the concatenated neighborhood is queried in a single forward pass.
    """
    ra_a, ra_b = _adjacent_ra_pair()
    ra_c, dec_c, pix_c = -6.0, -10.0, 67
    left = _catalog(pd.DataFrame({"id": [1, 2], "ra": [ra_a, ra_b], "dec": [0.0, 0.0]}), "left")
    right = _catalog(pd.DataFrame({"id": [10], "ra": [ra_c], "dec": [dec_c]}), "right")

    outer = left.crossmatch(
        right,
        how="outer",
        radius_arcsec=RADIUS,
        suffixes=("_left", "_right"),
        suffix_method="all_columns",
    )
    graph = outer._operation.build([HealpixPixel(ORDER, pix_c)])
    assert any("concat" in str(key) for key in graph.graph)

    result = pd.DataFrame(outer.compute())
    assert len(result) == 3
    assert set(result["id_left"].dropna()) == {1, 2}
    assert result[result["id_left"].isna()]["id_right"].tolist() == [10]


def test_outer_left_margin_excludes_matched_boundary_rows():
    """A right row matched to a left row across a left pixel boundary is emitted once.

    The right row is native to the left row's neighboring pixel; without the left
    margin cache it would be emitted both as a match (from the left row's pixel) and
    as unmatched (from its own pixel).
    """
    ra_near, ra_far = _adjacent_ra_pair()
    left = _catalog(
        pd.DataFrame({"id": [1, 2], "ra": [ra_near, round(ra_far + 0.005, 4)], "dec": [0.0, 0.0]}),
        "left",
    )
    right = _catalog(pd.DataFrame({"id": [10], "ra": [ra_far], "dec": [0.0]}), "right")

    result = pd.DataFrame(
        left.crossmatch(
            right,
            how="outer",
            radius_arcsec=RADIUS,
            suffixes=("_left", "_right"),
            suffix_method="all_columns",
        ).compute()
    )

    assert len(result) == 2
    matched = result[result["id_right"] == 10]
    assert len(matched) == 1
    assert matched["id_left"].tolist() == [1]
    assert result[result["id_left"] == 2]["id_right"].isna().all()


def _coarse_right_catalogs():
    ras = [0.0, 5.0, 10.0, 15.0]
    left = _catalog(pd.DataFrame({"id": range(4), "ra": ras, "dec": [0.0] * 4}), "left", order=3)
    right = _catalog(
        pd.DataFrame({"id": range(10, 14), "ra": [ra + 0.01 for ra in ras], "dec": [0.0] * 4}),
        "right",
        order=0,
    )
    return left, right


def test_outer_filters_coarse_right_partition_to_aligned_pixels():
    """A coarse right partition is not emitted once per finer left pixel."""
    left, right = _coarse_right_catalogs()

    result = pd.DataFrame(
        left.crossmatch(
            right,
            how="outer",
            radius_arcsec=1,
            suffixes=("_left", "_right"),
            suffix_method="all_columns",
        ).compute()
    )

    assert len(result) == 8
    assert set(result["id_left"].dropna()) == set(range(4))
    assert set(result["id_right"].dropna()) == set(range(10, 14))


def test_crossmatch_rejects_unknown_join_method(small_sky_catalog, small_sky_xmatch_catalog):
    with pytest.raises(ValueError, match="`how` needs"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, how="right")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("left_margin", "right_margin", "message"),
    [
        (None, 60, "left catalog to have a margin cache"),
        (RADIUS / 2, 60, "Left margin threshold"),
        (60, RADIUS * 1.5, "Right margin threshold"),
        (60, None, "right catalog to have a margin cache"),
    ],
)
def test_outer_requires_valid_margins(left_margin, right_margin, message):
    ra_a, ra_b = _adjacent_ra_pair()
    left = _catalog(
        pd.DataFrame({"id": [1], "ra": [ra_a], "dec": [0.0]}), "left", margin_threshold=left_margin
    )
    right = _catalog(
        pd.DataFrame({"id": [10], "ra": [ra_b], "dec": [0.0]}), "right", margin_threshold=right_margin
    )

    with pytest.raises(ValueError, match=message):
        left.crossmatch(right, how="outer", radius_arcsec=RADIUS)


def test_outer_requires_algorithm_radius(small_sky_catalog, small_sky_xmatch_catalog):
    class RadiuslessCrossmatch(AbstractCrossmatchAlgorithm):
        def perform_crossmatch(self, crossmatch_args):
            del crossmatch_args
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), pd.DataFrame()

    with pytest.raises(ValueError, match="radius_arcsec"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, how="outer", algorithm=RadiuslessCrossmatch())


def test_outer_rejects_mismatched_right_native_mask():
    """Outer assembly requires the native-right mask to align with the right dataframe."""
    alg = KdTreeCrossmatch(radius_arcsec=1)
    frame = npd.NestedFrame({"ra": [0.0], "dec": [0.0]})

    with pytest.raises(ValueError, match="native-right mask"):
        alg._create_crossmatch_df(
            frame,
            frame,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            pd.DataFrame(),
            how="outer",
            suffixes=("_l", "_r"),
            right_native_mask=np.array([True, True, True]),
        )


def test_outer_coarse_right_requires_spatial_index_order():
    """Filtering a coarse right partition to finer aligned pixels needs its spatial index."""
    left, right = _coarse_right_catalogs()
    right.hc_structure.catalog_info.healpix_order = None

    with pytest.raises(ValueError, match="spatial-index order"):
        left.crossmatch(
            right,
            how="outer",
            radius_arcsec=1,
            suffixes=("_left", "_right"),
            suffix_method="all_columns",
        ).compute()


def test_outer_supports_algorithms_without_extra_columns():
    class NoExtraColumnsCrossmatch(AbstractCrossmatchAlgorithm):
        radius_arcsec = 1.0

        def perform_crossmatch(self, crossmatch_args):
            del crossmatch_args
            return np.array([0]), np.array([0]), pd.DataFrame(index=[0])

    left_df = pd.DataFrame({"id": [1, 2], "ra": [0.0, 1.0], "dec": [0.0, 0.0]})
    right_df = pd.DataFrame({"id": [10, 11], "ra": [0.0, 2.0], "dec": [0.0, 0.0]})
    left = lsdb.from_dataframe(left_df, lowest_order=0, highest_order=0, margin_threshold=30)
    right = lsdb.from_dataframe(right_df, lowest_order=0, highest_order=0, margin_threshold=30)

    result = pd.DataFrame(
        left.crossmatch(
            right,
            how="outer",
            algorithm=NoExtraColumnsCrossmatch(),
            suffix_method="all_columns",
        ).compute()
    )

    assert len(result) == 3
