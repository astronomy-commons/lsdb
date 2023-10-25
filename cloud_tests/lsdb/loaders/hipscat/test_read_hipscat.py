import hipscat as hc
import pandas as pd
import pytest

import lsdb


def test_read_hipscat(
    small_sky_order1_dir_cloud, small_sky_order1_hipscat_catalog_cloud, example_cloud_storage_options
):
    catalog = lsdb.read_hipscat(small_sky_order1_dir_cloud, storage_options=example_cloud_storage_options)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hipscat_catalog_cloud.catalog_base_dir
    assert catalog.get_healpix_pixels() == small_sky_order1_hipscat_catalog_cloud.get_healpix_pixels()


def test_pixels_in_map_equal_catalog_pixels(
    small_sky_order1_dir_cloud, small_sky_order1_hipscat_catalog_cloud, example_cloud_storage_options
):
    catalog = lsdb.read_hipscat(small_sky_order1_dir_cloud, storage_options=example_cloud_storage_options)
    for healpix_pixel in small_sky_order1_hipscat_catalog_cloud.get_healpix_pixels():
        catalog.get_partition(healpix_pixel.order, healpix_pixel.pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir_cloud, example_cloud_storage_options):
    catalog = lsdb.read_hipscat(small_sky_order1_dir_cloud, storage_options=example_cloud_storage_options)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(
    small_sky_order1_dir_cloud, small_sky_order1_hipscat_catalog_cloud, example_cloud_storage_options
):
    catalog = lsdb.read_hipscat(small_sky_order1_dir_cloud, storage_options=example_cloud_storage_options)
    for healpix_pixel in small_sky_order1_hipscat_catalog_cloud.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hipscat_catalog_cloud.catalog_base_dir, hp_order, hp_pixel
        )
        loaded_df = pd.read_parquet(parquet_path, storage_options=example_cloud_storage_options)
        pd.testing.assert_frame_equal(partition_df, loaded_df)


def test_read_hipscat_specify_catalog_type(
    small_sky_catalog_cloud, small_sky_dir_cloud, example_cloud_storage_options
):
    catalog = lsdb.read_hipscat(
        small_sky_dir_cloud, catalog_type=lsdb.Catalog, storage_options=example_cloud_storage_options
    )
    assert isinstance(catalog, lsdb.Catalog)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog_cloud.compute())
    assert catalog.hc_structure.get_healpix_pixels() == small_sky_catalog_cloud.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog_cloud.hc_structure.catalog_info


def test_read_hipscat_no_parquet_metadata(
    small_sky_catalog_cloud, small_sky_no_metadata_dir_cloud, example_cloud_storage_options
):
    catalog = lsdb.read_hipscat(
        small_sky_no_metadata_dir_cloud, storage_options=example_cloud_storage_options
    )
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog_cloud.compute())
    assert catalog.hc_structure.get_healpix_pixels() == small_sky_catalog_cloud.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog_cloud.hc_structure.catalog_info


def test_read_hipscat_specify_wrong_catalog_type(small_sky_dir_cloud):
    with pytest.raises(ValueError):
        lsdb.read_hipscat(small_sky_dir_cloud, catalog_type=int)
