import logging

import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb
import lsdb.nested as nd
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from lsdb.dask.merge_catalog_functions import align_catalogs, apply_suffixes


def test_kdtree_crossmatch(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct, helpers):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched_cat = small_sky_catalog.crossmatch(small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600)
        assert isinstance(xmatched_cat._ddf, nd.NestedFrame)
        xmatched = xmatched_cat.compute()
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert xmatched_cat.hc_structure.moc == alignment.moc
    assert xmatched_cat.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()
    helpers.assert_schema_correct(xmatched_cat)

    assert isinstance(xmatched, npd.NestedFrame)
    assert len(xmatched) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)

    assert xmatched_cat.hc_structure.catalog_info.ra_column in xmatched_cat.columns
    assert xmatched_cat.hc_structure.catalog_info.dec_column in xmatched_cat.columns
    assert xmatched_cat.hc_structure.catalog_info.ra_column == "ra_small_sky"
    assert xmatched_cat.hc_structure.catalog_info.dec_column == "dec_small_sky"


def test_kdtree_crossmatch_nested(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched_cat = small_sky_catalog.crossmatch_nested(
            small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600
        )
        assert isinstance(xmatched_cat._ddf, nd.NestedFrame)
        xmatched = xmatched_cat.compute()
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert xmatched_cat.hc_structure.moc == alignment.moc
    assert xmatched_cat.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    assert isinstance(xmatched, npd.NestedFrame)
    assert np.sum(xmatched["small_sky_xmatch"].nest.list_lengths) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in xmatched["id"].to_numpy()
        xmatch_row = xmatched[xmatched["id"] == correct_row["ss_id"]]
        assert xmatch_row["small_sky_xmatch"].iloc[0]["id"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["small_sky_xmatch"].iloc[0]["_dist_arcsec"].to_numpy() == pytest.approx(
            correct_row["dist"] * 3600
        )


def test_kdtree_crossmatch_nested_custom_name(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct):
    nested_column_name = "xmatches"
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        cat_name = "xmatched_cat"
        xmatched_cat = small_sky_catalog.crossmatch_nested(
            small_sky_xmatch_catalog,
            radius_arcsec=0.01 * 3600,
            nested_column_name=nested_column_name,
            output_catalog_name=cat_name,
        )
        assert isinstance(xmatched_cat._ddf, nd.NestedFrame)
        assert xmatched_cat.name == cat_name
        xmatched = xmatched_cat.compute()
    alignment = align_catalogs(small_sky_catalog, small_sky_xmatch_catalog)
    assert xmatched_cat.hc_structure.moc == alignment.moc
    assert xmatched_cat.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()

    assert isinstance(xmatched, npd.NestedFrame)
    assert np.sum(xmatched[nested_column_name].nest.list_lengths) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in xmatched["id"].to_numpy()
        xmatch_row = xmatched[xmatched["id"] == correct_row["ss_id"]]
        assert xmatch_row[nested_column_name].iloc[0]["id"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row[nested_column_name].iloc[0]["_dist_arcsec"].to_numpy() == pytest.approx(
            correct_row["dist"] * 3600
        )


def test_kdtree_crossmatch_default_cols(
    small_sky_order1_default_cols_catalog, small_sky_xmatch_catalog, xmatch_correct, helpers
):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched_cat = small_sky_order1_default_cols_catalog.crossmatch(
            small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600
        )
        assert isinstance(xmatched_cat._ddf, nd.NestedFrame)
        xmatched = xmatched_cat.compute()
    helpers.assert_schema_correct(xmatched_cat)
    helpers.assert_default_columns_in_columns(xmatched_cat)
    alignment = align_catalogs(small_sky_order1_default_cols_catalog, small_sky_xmatch_catalog)
    assert xmatched_cat.hc_structure.moc == alignment.moc
    assert xmatched_cat.get_healpix_pixels() == alignment.pixel_tree.get_healpix_pixels()
    assert not xmatched_cat.hc_structure.on_disk

    assert isinstance(xmatched, npd.NestedFrame)
    assert len(xmatched) == len(xmatch_correct)
    for _, correct_row in xmatch_correct.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky_order1_default_columns"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky_order1_default_columns"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


def test_kdtree_crossmatch_thresh(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            radius_arcsec=0.005 * 3600,
        ).compute()
    assert len(xmatched) == len(xmatch_correct_005)
    for _, correct_row in xmatch_correct_005.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


def test_kdtree_crossmatch_multiple_neighbors(
    small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t_no_margin
):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            n_neighbors=3,
            radius_arcsec=2 * 3600,
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


def test_kdtree_crossmatch_nested_multiple_neighbors(
    small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_3n_2t_no_margin
):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch_nested(
            small_sky_xmatch_catalog,
            n_neighbors=3,
            radius_arcsec=2 * 3600,
        ).compute()
    assert np.sum(xmatched["small_sky_xmatch"].nest.list_lengths) == len(xmatch_correct_3n_2t_no_margin)
    for _, correct_row in xmatch_correct_3n_2t_no_margin.iterrows():
        assert correct_row["ss_id"] in xmatched["id"].to_numpy()
        xmatch_df = xmatched[xmatched["id"] == correct_row["ss_id"]]["small_sky_xmatch"].iloc[0]
        xmatch_row = xmatch_df[xmatch_df["id"] == correct_row["xmatch_id"]]
        assert len(xmatch_row) == 1
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


def test_kdtree_crossmatch_multiple_neighbors_margin(
    small_sky_catalog, small_sky_xmatch_dir, small_sky_xmatch_margin_dir, xmatch_correct_3n_2t
):
    small_sky_xmatch_catalog = lsdb.open_catalog(
        small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
    )
    xmatched = small_sky_catalog.crossmatch(
        small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600
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


def test_crossmatch_negative_margin(
    small_sky_left_xmatch_catalog,
    small_sky_xmatch_dir,
    small_sky_xmatch_margin_dir,
    xmatch_correct_3n_2t_negative,
):
    small_sky_xmatch_catalog = lsdb.open_catalog(
        small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
    )
    xmatched = small_sky_left_xmatch_catalog.crossmatch(
        small_sky_xmatch_catalog, n_neighbors=3, radius_arcsec=2 * 3600
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


def test_overlapping_suffix_method(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    suffixes = ("_left", "_right")
    # Test that renamed columns are logged correctly
    with caplog.at_level(logging.WARNING):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            suffix_method="overlapping_columns",
            suffixes=suffixes,
        )

        assert caplog.text.count("Renaming overlapping columns") == 1

        computed = xmatched.compute()

        assert caplog.text.count("Renaming overlapping columns") == 1

    for col in small_sky_catalog.columns:
        if col in small_sky_xmatch_catalog.columns:
            assert f"{col}{suffixes[0]}" in xmatched.columns
            assert f"{col}{suffixes[0]}" in computed.columns
            assert col in caplog.text
            assert f"{col}{suffixes[0]}" in caplog.text
        else:
            assert col in xmatched.columns
            assert col in computed.columns
    for col in small_sky_xmatch_catalog.columns:
        if col in small_sky_catalog.columns:
            assert f"{col}{suffixes[1]}" in xmatched.columns
            assert f"{col}{suffixes[1]}" in computed.columns
            assert f"{col}{suffixes[1]}" in caplog.text
        else:
            assert col in xmatched.columns
            assert col in computed.columns

    assert xmatched.hc_structure.catalog_info.ra_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.dec_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.ra_column == "ra_left"
    assert xmatched.hc_structure.catalog_info.dec_column == "dec_left"


def test_overlapping_suffix_method_no_overlaps(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    suffixes = ("_left", "_right")
    small_sky_catalog = small_sky_catalog.rename({col: f"{col}_unique" for col in small_sky_catalog.columns})
    small_sky_catalog.hc_structure.catalog_info.ra_column = (
        f"{small_sky_catalog.hc_structure.catalog_info.ra_column}_unique"
    )
    small_sky_catalog.hc_structure.catalog_info.dec_column = (
        f"{small_sky_catalog.hc_structure.catalog_info.dec_column}_unique"
    )
    # Test that renamed columns are logged correctly
    with caplog.at_level(logging.INFO):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            suffix_method="overlapping_columns",
            suffixes=suffixes,
        )

    assert len(caplog.text) == 0

    computed = xmatched.compute()
    for col in small_sky_catalog.columns:
        assert col in xmatched.columns
        assert col in computed.columns
    for col in small_sky_xmatch_catalog.columns:
        assert col in xmatched.columns
        assert col in computed.columns

    assert xmatched.hc_structure.catalog_info.ra_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.dec_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.ra_column == "ra_unique"
    assert xmatched.hc_structure.catalog_info.dec_column == "dec_unique"


def test_overlapping_suffix_log_changes_false(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    suffixes = ("_left", "_right")
    # Test that renamed columns are not logged correctly
    with caplog.at_level(logging.WARNING):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            suffix_method="overlapping_columns",
            suffixes=suffixes,
            log_changes=False,
        )

        assert len(caplog.text) == 0
        computed = xmatched.compute()
        assert len(caplog.text) == 0

    for col in small_sky_catalog.columns:
        if col in small_sky_xmatch_catalog.columns:
            assert f"{col}{suffixes[0]}" in xmatched.columns
            assert f"{col}{suffixes[0]}" in computed.columns
        else:
            assert col in xmatched.columns
            assert col in computed.columns
    for col in small_sky_xmatch_catalog.columns:
        if col in small_sky_catalog.columns:
            assert f"{col}{suffixes[1]}" in xmatched.columns
            assert f"{col}{suffixes[1]}" in computed.columns
        else:
            assert col in xmatched.columns
            assert col in computed.columns

    assert xmatched.hc_structure.catalog_info.ra_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.dec_column in xmatched.columns
    assert xmatched.hc_structure.catalog_info.ra_column == "ra_left"
    assert xmatched.hc_structure.catalog_info.dec_column == "dec_left"


def test_wrong_suffixes(small_sky_catalog, small_sky_xmatch_catalog):
    with pytest.raises(ValueError):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, suffixes=("wrong",))


def test_wrong_suffix_method(small_sky_catalog, small_sky_xmatch_catalog):
    with pytest.raises(ValueError, match="Invalid suffix method"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, suffix_method="wrong")


def test_right_margin_missing(small_sky_catalog, small_sky_xmatch_catalog):
    small_sky_xmatch_catalog.margin = None
    with pytest.raises(ValueError, match="Right catalog margin"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, require_right_margin=True)


def test_kdtree_crossmatch_min_thresh(small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_002_005):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            radius_arcsec=0.005 * 3600,
            min_radius_arcsec=0.002 * 3600,
        ).compute()
    assert len(xmatched) == len(xmatch_correct_002_005)
    for _, correct_row in xmatch_correct_002_005.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


def test_kdtree_crossmatch_min_thresh_multiple_neighbors_margin(
    small_sky_catalog,
    small_sky_xmatch_dir,
    small_sky_xmatch_margin_dir,
    xmatch_correct_05_2_3n_margin,
):
    small_sky_xmatch_catalog = lsdb.open_catalog(
        small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir
    )
    xmatched = small_sky_catalog.crossmatch(
        small_sky_xmatch_catalog,
        n_neighbors=3,
        radius_arcsec=2 * 3600,
        min_radius_arcsec=0.5 * 3600,
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


def test_kdtree_crossmatch_no_close_neighbors(
    small_sky_catalog, small_sky_xmatch_catalog, xmatch_correct_005
):
    # Set a very small minimum radius so that there is not a single point
    # with a very close neighbor
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            radius_arcsec=0.005 * 3600,
            min_radius_arcsec=1,
        ).compute()
    assert len(xmatched) == len(xmatch_correct_005)
    for _, correct_row in xmatch_correct_005.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["dist"] * 3600)


def test_crossmatch_more_neighbors_than_points_available(small_sky_catalog, small_sky_xmatch_catalog):
    # The small_sky_xmatch catalog has 3 partitions (2 of length 41 and 1 of length 29).
    # Let's use n_neighbors above that to request more neighbors than there are points available.
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog,
            n_neighbors=50,
            radius_arcsec=2 * 3600,
            min_radius_arcsec=0.5 * 3600,
        ).compute()
    assert len(xmatched) == 72
    assert all(xmatched.groupby("id_small_sky").size()) <= 50


def test_self_crossmatch(small_sky_catalog, small_sky_dir):
    # Read a second small sky catalog to not have duplicate labels
    small_sky_catalog_2 = lsdb.open_catalog(small_sky_dir)
    small_sky_catalog_2.hc_structure.catalog_name = "small_sky_2"
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_catalog_2,
            min_radius_arcsec=0,
            radius_arcsec=0.005 * 3600,
        ).compute()
    assert len(xmatched) == len(small_sky_catalog.compute())
    assert all(xmatched["_dist_arcsec"] == 0)


def test_crossmatch_empty_left_partition(small_sky_order1_catalog, small_sky_xmatch_catalog):
    ra = 300
    dec = -60
    radius_arcsec = 3 * 3600
    cone = small_sky_order1_catalog.cone_search(ra, dec, radius_arcsec)
    assert len(cone.get_healpix_pixels()) == 2
    assert len(cone.get_partition(1, 44)) == 5
    # There is an empty partition in the left catalog
    assert len(cone.get_partition(1, 46)) == 0
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = cone.crossmatch(small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600).compute()
    assert len(xmatched) == 3
    assert all(xmatched["_dist_arcsec"] <= 0.01 * 3600)


def test_crossmatch_empty_right_partition(small_sky_order1_catalog, small_sky_xmatch_catalog):
    ra = 300
    dec = -60
    radius_arcsec = 3.4 * 3600
    cone = small_sky_xmatch_catalog.cone_search(ra, dec, radius_arcsec)
    assert len(cone.get_healpix_pixels()) == 2
    assert len(cone.get_partition(1, 44)) == 5
    # There is an empty partition in the right catalog
    assert len(cone.get_partition(1, 46)) == 0
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_order1_catalog.crossmatch(cone, radius_arcsec=0.01 * 3600).compute()
    assert len(xmatched) == 3
    assert all(xmatched["_dist_arcsec"] <= 0.01 * 3600)


def test_kdtree_crossmatch_left_join_preserves_left_rows(small_sky_order1_catalog, small_sky_xmatch_catalog):
    """Check that a left join preserves left rows and keeps valid matches."""
    radius = 0.01 * 3600
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        inner = small_sky_order1_catalog.crossmatch(small_sky_xmatch_catalog, radius_arcsec=radius).compute()
        left_joined = small_sky_order1_catalog.crossmatch(
            small_sky_xmatch_catalog, radius_arcsec=radius, how="left"
        ).compute()

    # Every left id from the left catalog subset should appear at least once in the left-joined result
    left_ids = small_sky_order1_catalog.compute()["id"].to_numpy()
    # Using .issubset as a good practice in case of multiple matches
    assert set(left_ids).issubset(set(left_joined["id_small_sky_order1"].to_numpy()))

    # Every (left_id, right_id) pair present in the inner join
    # must also be present in the left-join result (i.e. left-join doesn't drop inner matches).
    inner_pairs = set(zip(inner["id_small_sky_order1"].to_numpy(), inner["id_small_sky_xmatch"].to_numpy()))
    left_pairs = set(
        zip(left_joined["id_small_sky_order1"].to_numpy(), left_joined["id_small_sky_xmatch"].to_numpy())
    )
    assert inner_pairs.issubset(left_pairs)

    # Matched rows should have distances within the threshold
    matched = left_joined[left_joined["id_small_sky_xmatch"].notna()]
    assert all(matched["_dist_arcsec"] <= radius)

    # Unmatched rows should have NA in the right id column
    unmatched = left_joined[left_joined["id_small_sky_xmatch"].isna()]
    assert len(unmatched) >= 0

    # No rows should have NA in the index, either in left_joined or inner
    assert len(inner[inner.index.isna()]["id_small_sky_xmatch"]) == 0
    assert len(left_joined[left_joined.index.isna()]["id_small_sky_xmatch"]) == 0


def test_kdtree_crossmatch_left_join_non_unique_left_index(
    small_sky_order1_catalog, small_sky_xmatch_catalog
):
    """Check that left join handles non-unique values in the left index correctly.

    This tests the workaround for handling non-unique index values, which can occur
    when using `.iloc[indices]` on DataFrames where the original index has duplicates.
    The code should correctly identify unmatched rows even when multiple rows share
    the same index value.
    """
    # Create a left catalog with deliberately non-unique index values
    left_ddf = small_sky_order1_catalog._ddf
    # Duplicate the index to create non-unique values
    left_ddf_dup_index = left_ddf.copy()
    left_ddf_dup_index.index = left_ddf_dup_index.index.map(lambda x: x if x % 2 == 0 else 0)
    small_sky_order1_catalog_dup_idx = small_sky_order1_catalog._create_updated_dataset(
        ddf=left_ddf_dup_index
    )

    radius = 0.01 * 3600
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        left_joined = small_sky_order1_catalog_dup_idx.crossmatch(
            small_sky_xmatch_catalog, radius_arcsec=radius, how="left"
        ).compute()

    # Verify result has rows (both matched and unmatched)
    assert len(left_joined) > 0

    # Verify that matched rows have valid distance values
    matched = left_joined[left_joined["id_small_sky_xmatch"].notna()]
    if len(matched) > 0:
        assert all(matched["_dist_arcsec"] <= radius)

    # Verify that unmatched rows have NaN in right columns
    unmatched = left_joined[left_joined["id_small_sky_xmatch"].isna()]
    for col in left_joined.columns:
        if col.endswith("_small_sky_xmatch"):
            # Right-side columns should be NaN for unmatched rows
            assert unmatched[col].isna().all() or (
                pd.api.types.is_object_dtype(unmatched[col]) and unmatched[col].isna().all()
            )

    # Ensure extra columns (distance) are NaN for unmatched rows
    unmatched_extra = unmatched[unmatched.columns.intersection(["_dist_arcsec"])]
    if len(unmatched_extra.columns) > 0:
        assert unmatched_extra.isna().all().all()


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


def test_crossmatch_with_non_catalog(small_sky_catalog, small_sky_xmatch_catalog):
    # Turn the small_sky_xmatch_catalog into a DataFrame
    xmatch_df = small_sky_xmatch_catalog.compute()
    # Perform crossmatch with a DataFrame instead of a Catalog
    with pytest.raises(TypeError, match="Expected `other` to be a Catalog instance"):
        small_sky_catalog.crossmatch(xmatch_df)


# pylint: disable=too-few-public-methods, unused-argument
class MockCrossmatchAlgorithm(AbstractCrossmatchAlgorithm):
    """Mock class used to test a crossmatch algorithm"""

    extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=np.float64)})

    def __init__(self, mock_results: pd.DataFrame = None, n_neighbors: int = 1):
        self.mock_results = mock_results
        self.n_neighbors = n_neighbors

    def perform_crossmatch(self, crossmatch_args: CrossmatchArgs):
        left_reset = crossmatch_args.left_df.reset_index(drop=True)
        right_reset = crossmatch_args.right_df.reset_index(drop=True)
        mock_results = self.mock_results[self.mock_results["ss_id"].isin(left_reset["id"].to_numpy())]
        left_indexes = mock_results.apply(
            lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1
        )
        right_indexes = mock_results.apply(
            lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1
        )
        extra_columns = pd.DataFrame({"_DIST": mock_results["dist"]})

        return left_indexes.to_numpy(), right_indexes.to_numpy(), extra_columns


def test_custom_crossmatch_algorithm(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithm(mock_results=xmatch_mock)
        ).compute()
    assert len(xmatched) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id_small_sky"].to_numpy()
        xmatch_row = xmatched[xmatched["id_small_sky"] == correct_row["ss_id"]]
        assert xmatch_row["id_small_sky_xmatch"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].to_numpy() == pytest.approx(correct_row["dist"])


def test_custom_crossmatch_algorithm_nested(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch_nested(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithm(mock_results=xmatch_mock)
        ).compute()
    assert np.sum(xmatched["small_sky_xmatch"].nest.list_lengths) == len(xmatch_mock)
    for _, correct_row in xmatch_mock.iterrows():
        assert correct_row["ss_id"] in xmatched["id"].to_numpy()
        xmatch_row = xmatched[xmatched["id"] == correct_row["ss_id"]]["small_sky_xmatch"].iloc[0]
        assert xmatch_row["id"].to_numpy() == correct_row["xmatch_id"]
        assert xmatch_row["_DIST"].to_numpy() == pytest.approx(correct_row["dist"])


def test_custom_crossmatch_algorithm_with_default_kwargs(
    small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock
):
    # If the custom crossmatch algorithm has attributes whose names conflict
    # with those of the default `KdTreeCrossmatch` kwargs, they need to be
    # provided in the crossmatch algorithm constructor.
    algorithm = MockCrossmatchAlgorithm(mock_results=xmatch_mock)
    with pytest.raises(ValueError, match="do not set"):
        small_sky_catalog.crossmatch(small_sky_xmatch_catalog, algorithm=algorithm, n_neighbors=2)
    with pytest.raises(ValueError, match="do not set"):
        small_sky_catalog.crossmatch_nested(small_sky_xmatch_catalog, algorithm=algorithm, n_neighbors=2)
    algorithm = MockCrossmatchAlgorithm(mock_results=xmatch_mock, n_neighbors=2)
    small_sky_catalog.crossmatch(small_sky_xmatch_catalog, algorithm=algorithm)


# pylint: disable=too-many-arguments, abstract-method
class MockCrossmatchAlgorithmOverwrite(AbstractCrossmatchAlgorithm):
    """Mock class used to test a crossmatch algorithm"""

    extra_columns = pd.DataFrame({"_DIST": pd.Series(dtype=np.float64)})

    def __init__(self, mock_results: pd.DataFrame = None):
        self.mock_results = mock_results

    def crossmatch(self, crossmatch_args, how, suffixes, suffix_method="all_columns"):
        left_reset = crossmatch_args.left_df.reset_index(drop=True)
        right_reset = crossmatch_args.right_df.reset_index(drop=True)
        left, right = apply_suffixes(
            crossmatch_args.left_df,
            crossmatch_args.right_df,
            suffixes,
            suffix_method,
        )
        mock_results = self.mock_results[self.mock_results["ss_id"].isin(left_reset["id"].to_numpy())]
        left_indexes = mock_results.apply(
            lambda row: left_reset[left_reset["id"] == row["ss_id"]].index[0], axis=1
        )
        right_indexes = mock_results.apply(
            lambda row: right_reset[right_reset["id"] == row["xmatch_id"]].index[0], axis=1
        )
        left_join_part = left.iloc[left_indexes.to_numpy()].reset_index()
        right_join_part = right.iloc[right_indexes.to_numpy()].reset_index(drop=True)
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


def test_custom_crossmatch_algorithm_overwrite(small_sky_catalog, small_sky_xmatch_catalog, xmatch_mock):
    with pytest.warns(RuntimeWarning, match="Results may be incomplete and/or inaccurate"):
        xmatched = small_sky_catalog.crossmatch(
            small_sky_xmatch_catalog, algorithm=MockCrossmatchAlgorithmOverwrite(mock_results=xmatch_mock)
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


def test_raise_for_non_overlapping_catalogs(small_sky_order1_catalog, small_sky_xmatch_catalog):
    small_sky_order1_catalog = small_sky_order1_catalog.pixel_search([HealpixPixel(1, 44)])
    small_sky_xmatch_catalog = small_sky_xmatch_catalog.pixel_search([HealpixPixel(1, 45)])
    with pytest.raises(RuntimeError, match="overlap"):
        small_sky_order1_catalog.crossmatch(small_sky_xmatch_catalog)


def test_crossmatch_alignment_pixel_left_join_filters_and_aligns():
    left_df = pd.DataFrame({"id": [1, 2], "ra": [0.0, 0.01], "dec": [0.0, 0.01]})
    left_df["id"] = left_df["id"].astype(pd.ArrowDtype(pa.int64()))
    right_df = pd.DataFrame({"id": [101.0, 202.0], "ra": [0.0005, 0.2], "dec": [0.0, 0.2]})

    left_catalog = lsdb.from_dataframe(
        left_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=1,
        highest_order=1,
        margin_order=2,
        margin_threshold=30,
    )
    right_catalog = lsdb.from_dataframe(
        right_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=3,
        highest_order=3,
        margin_order=4,
        margin_threshold=30,
    )

    xmatched = left_catalog.crossmatch(
        right_catalog,
        how="left",
        radius_arcsec=10,
        suffixes=("_l", "_r"),
    )
    result = xmatched.compute()

    assert len(result) == len(left_df)
    assert not result.index.duplicated().any()

    matched = result[result["id_l"] == 1]
    assert len(matched) == 1
    assert matched["id_r"].iloc[0] == pytest.approx(101.0)
    assert matched["_dist_arcsec"].iloc[0] <= 10

    unmatched = result[result["id_l"] == 2]
    assert unmatched["id_r"].isna().all()

    left_orders = {p.order for p in left_catalog.get_healpix_pixels()}
    right_orders = {p.order for p in right_catalog.get_healpix_pixels()}
    aligned_orders = {p.order for p in xmatched.get_healpix_pixels()}
    assert aligned_orders
    assert max(aligned_orders) >= max(right_orders)
    assert min(aligned_orders) > min(left_orders)
