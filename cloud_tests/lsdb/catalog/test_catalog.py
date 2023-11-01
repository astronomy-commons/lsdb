import dask.dataframe as dd
import pandas as pd
from hipscat.pixel_math import HealpixPixel


def test_catalog_repr_equals_ddf_repr(small_sky_order1_catalog_cloud):
    assert repr(small_sky_order1_catalog_cloud) == repr(small_sky_order1_catalog_cloud._ddf)


def test_catalog_html_repr_equals_ddf_html_repr(small_sky_order1_catalog_cloud):
    assert small_sky_order1_catalog_cloud._repr_html_() == small_sky_order1_catalog_cloud._ddf._repr_html_()


def test_catalog_compute_equals_ddf_compute(small_sky_order1_catalog_cloud):
    pd.testing.assert_frame_equal(
        small_sky_order1_catalog_cloud.compute(), small_sky_order1_catalog_cloud._ddf.compute()
    )


def test_get_catalog_partition_gets_correct_partition(small_sky_order1_catalog_cloud):
    for pixel in small_sky_order1_catalog_cloud.hc_structure.get_healpix_pixels:
        hp_order = pixel.order
        hp_pixel = pixel.pixel
        partition = small_sky_order1_catalog_cloud.get_partition(hp_order, hp_pixel)
        pixel = HealpixPixel(order=hp_order, pixel=hp_pixel)
        partition_index = small_sky_order1_catalog_cloud._ddf_pixel_map[pixel]
        ddf_partition = small_sky_order1_catalog_cloud._ddf.partitions[partition_index]
        dd.utils.assert_eq(partition, ddf_partition)
