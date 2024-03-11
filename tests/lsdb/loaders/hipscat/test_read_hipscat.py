import hipscat as hc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import lsdb
from lsdb.core.search import ConeSearch


def test_read_hipscat(small_sky_order1_dir, small_sky_order1_hipscat_catalog, assert_divisions_are_correct):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hipscat_catalog.catalog_base_dir
    assert catalog.get_healpix_pixels() == small_sky_order1_hipscat_catalog.get_healpix_pixels()
    assert len(catalog.compute().columns) == 8
    assert_divisions_are_correct(catalog)


def test_read_hipscat_with_columns(small_sky_order1_dir):
    filter_columns = ["ra", "dec"]
    catalog = lsdb.read_hipscat(small_sky_order1_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    npt.assert_array_equal(catalog.compute().columns.values, filter_columns)


def test_read_hipscat_with_extra_kwargs(small_sky_order1_dir):
    catalog = lsdb.read_hipscat(small_sky_order1_dir, filters=[("ra", ">", 300)], engine="pyarrow")
    assert isinstance(catalog, lsdb.Catalog)
    assert np.greater(catalog.compute()["ra"].values, 300).all()


def test_pixels_in_map_equal_catalog_pixels(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hipscat_catalog.get_healpix_pixels():
        catalog.get_partition(healpix_pixel.order, healpix_pixel.pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
    catalog = lsdb.read_hipscat(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hipscat_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hipscat_catalog.catalog_base_dir, hp_order, hp_pixel
        )
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(partition_df, loaded_df)


def test_read_hipscat_specify_catalog_type(small_sky_catalog, small_sky_dir):
    catalog = lsdb.read_hipscat(small_sky_dir, catalog_type=lsdb.Catalog)
    assert isinstance(catalog, lsdb.Catalog)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog.compute())
    assert catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info


def test_read_hipscat_specify_wrong_catalog_type(small_sky_dir):
    with pytest.raises(ValueError):
        lsdb.read_hipscat(small_sky_dir, catalog_type=int)


def test_catalog_with_margin(small_sky_xmatch_dir, small_sky_xmatch_margin_catalog):
    catalog = lsdb.read_hipscat(small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_catalog)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.margin is small_sky_xmatch_margin_catalog


def test_catalog_without_margin_is_none(small_sky_xmatch_dir):
    catalog = lsdb.read_hipscat(small_sky_xmatch_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.margin is None


def test_read_hipscat_subset(
    small_sky_order1_dir, small_sky_order1_hipscat_catalog, assert_divisions_are_correct
):
    catalog = lsdb.read_hipscat_subset(small_sky_order1_dir, order=1, n_pixels=2)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hipscat_catalog.catalog_base_dir
    assert len(catalog.get_healpix_pixels()) == 2
    assert catalog.hc_structure.catalog_info.total_rows is None
    assert_divisions_are_correct(catalog)


def test_read_hipscat_subset_with_cone_search(
    small_sky_order1_dir, small_sky_order1_hipscat_catalog, assert_divisions_are_correct
):
    # TODO: Instead of creating filter like this, allow user to pass the params directly to function
    cone_search = ConeSearch(ra=0, dec=-80, radius_arcsec=20 * 3600, metadata=None)
    catalog = lsdb.read_hipscat_subset(small_sky_order1_dir, search_filter=cone_search)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hipscat_catalog.catalog_base_dir
    assert len(catalog.get_healpix_pixels()) == 2
    assert catalog.hc_structure.catalog_info.total_rows is None
    assert_divisions_are_correct(catalog)


def test_read_hipscat_subset_warns_few_pixels_at_order(small_sky_order1_dir, assert_divisions_are_correct):
    with pytest.warns(RuntimeWarning, match="less than"):
        catalog = lsdb.read_hipscat_subset(small_sky_order1_dir, order=1, n_pixels=10)
    assert isinstance(catalog, lsdb.Catalog)
    assert len(catalog.get_healpix_pixels()) == 4
    assert_divisions_are_correct(catalog)
