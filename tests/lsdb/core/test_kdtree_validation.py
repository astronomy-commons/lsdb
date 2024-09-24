"""Tests on dataframe and argument validation on the default KD Tree crossmatch."""

import pytest

from lsdb.core.crossmatch.bounded_kdtree_match import BoundedKdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


def test_kdtree_radius_invalid(small_sky_catalog, small_sky_order1_source_with_margin):
    with pytest.raises(ValueError, match="radius must be positive"):
        KdTreeCrossmatch.validate(small_sky_catalog, small_sky_order1_source_with_margin, radius_arcsec=-36)
    with pytest.raises(ValueError, match="n_neighbors"):
        KdTreeCrossmatch.validate(small_sky_catalog, small_sky_order1_source_with_margin, n_neighbors=0)
    with pytest.raises(ValueError, match="Cross match radius is greater"):
        KdTreeCrossmatch.validate(
            small_sky_catalog, small_sky_order1_source_with_margin, radius_arcsec=3 * 3600
        )


def test_bounded_kdtree_radius_invalid(small_sky_catalog, small_sky_order1_source_with_margin):
    with pytest.raises(ValueError, match="radius must be non-negative"):
        BoundedKdTreeCrossmatch.validate(
            small_sky_catalog, small_sky_order1_source_with_margin, min_radius_arcsec=-36
        )
    with pytest.raises(ValueError, match="maximum radius must be greater than"):
        BoundedKdTreeCrossmatch.validate(
            small_sky_catalog, small_sky_order1_source_with_margin, min_radius_arcsec=2, radius_arcsec=1
        )


def test_kdtree_left_columns(small_sky_catalog, small_sky_order1_source_with_margin):
    original_df = small_sky_catalog._ddf
    small_sky_catalog._ddf = original_df.reset_index()
    with pytest.raises(ValueError, match="index of left table must be _healpix_29"):
        KdTreeCrossmatch.validate(
            small_sky_catalog,
            small_sky_order1_source_with_margin,
        )

    small_sky_catalog._ddf = original_df.drop(columns=["ra"])
    with pytest.raises(ValueError, match="left table must have column ra"):
        KdTreeCrossmatch.validate(
            small_sky_catalog,
            small_sky_order1_source_with_margin,
        )

    small_sky_catalog._ddf = original_df.drop(columns=["dec"])
    with pytest.raises(ValueError, match="left table must have column dec"):
        KdTreeCrossmatch.validate(
            small_sky_catalog,
            small_sky_order1_source_with_margin,
        )


def test_kdtree_right_columns(small_sky_catalog, small_sky_order1_source_with_margin):
    original_df = small_sky_order1_source_with_margin._ddf
    small_sky_order1_source_with_margin._ddf = original_df.reset_index()
    with pytest.raises(ValueError, match="index of right table must be _healpix_29"):
        KdTreeCrossmatch.validate(small_sky_catalog, small_sky_order1_source_with_margin)

    small_sky_order1_source_with_margin._ddf = original_df.drop(columns=["source_ra"])
    with pytest.raises(ValueError, match="right table must have column source_ra"):
        KdTreeCrossmatch.validate(small_sky_catalog, small_sky_order1_source_with_margin)

    small_sky_order1_source_with_margin._ddf = original_df.drop(columns=["source_dec"])
    with pytest.raises(ValueError, match="right table must have column source_dec"):
        KdTreeCrossmatch.validate(small_sky_catalog, small_sky_order1_source_with_margin)
