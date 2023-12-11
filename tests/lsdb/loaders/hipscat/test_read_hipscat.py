import hipscat as hc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from conftest import assert_divisions_are_correct  # pylint: disable=import-error

import lsdb


def test_read_hipscat(small_sky_order1_dir, small_sky_order1_hipscat_catalog):
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
