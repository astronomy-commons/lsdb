"""Tests on dataframe and argument validation on the default KD Tree crossmatch."""

import pytest

from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


@pytest.fixture
def kdtree_crossmatch(small_sky_catalog, small_sky_order1_source_with_margin):
    algo = KdTreeCrossmatch(
        small_sky_catalog._ddf,
        small_sky_order1_source_with_margin._ddf,
        0,
        0,
        0,
        0,
        small_sky_catalog.hc_structure,
        small_sky_order1_source_with_margin.hc_structure,
        small_sky_order1_source_with_margin.margin.hc_structure,
        ("_a", "_b"),
    )
    return algo


def test_kdtree_radius_invalid(kdtree_crossmatch):
    with pytest.raises(ValueError, match="radius must be positive"):
        kdtree_crossmatch.validate(radius_arcsec=-36)
    with pytest.raises(ValueError, match="n_neighbors"):
        kdtree_crossmatch.validate(n_neighbors=0)

    with pytest.raises(ValueError, match="Cross match radius is greater"):
        kdtree_crossmatch.validate(radius_arcsec=10 * 3600)


def test_kdtree_no_margin(kdtree_crossmatch):
    kdtree_crossmatch.right_margin_hc_structure = None
    with pytest.raises(ValueError, match="Right margin is required"):
        kdtree_crossmatch.validate()

    kdtree_crossmatch.validate(require_right_margin=False)


def test_kdtree_left_columns(kdtree_crossmatch):
    original_df = kdtree_crossmatch.left
    kdtree_crossmatch.left = original_df.reset_index()
    with pytest.raises(ValueError, match="index of left table must be _hipscat_index"):
        kdtree_crossmatch.validate()

    kdtree_crossmatch.left = original_df.drop(columns=["ra"])
    with pytest.raises(ValueError, match="left table must have column ra"):
        kdtree_crossmatch.validate()

    kdtree_crossmatch.left = original_df.drop(columns=["dec"])
    with pytest.raises(ValueError, match="left table must have column dec"):
        kdtree_crossmatch.validate()


def test_kdtree_right_columns(kdtree_crossmatch):
    original_df = kdtree_crossmatch.right
    kdtree_crossmatch.right = original_df.reset_index()
    with pytest.raises(ValueError, match="index of right table must be _hipscat_index"):
        kdtree_crossmatch.validate()

    kdtree_crossmatch.right = original_df.drop(columns=["source_ra"])
    with pytest.raises(ValueError, match="right table must have column source_ra"):
        kdtree_crossmatch.validate()

    kdtree_crossmatch.right = original_df.drop(columns=["source_dec"])
    with pytest.raises(ValueError, match="right table must have column source_dec"):
        kdtree_crossmatch.validate()
