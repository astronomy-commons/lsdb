import dask.dataframe as dd
import pandas as pd
import pytest
from hipscat.pixel_math import HealpixPixel


def test_catalog_pixels_equals_hc_catalog_pixels(small_sky_order1_catalog, small_sky_order1_hipscat_catalog):
    assert (
        small_sky_order1_catalog.get_healpix_pixels() == small_sky_order1_hipscat_catalog.get_healpix_pixels()
    )


def test_catalog_repr_equals_ddf_repr(small_sky_order1_catalog):
    assert repr(small_sky_order1_catalog) == repr(small_sky_order1_catalog._ddf)


def test_catalog_html_repr_equals_ddf_html_repr(small_sky_order1_catalog):
    assert small_sky_order1_catalog._repr_html_() == small_sky_order1_catalog._ddf._repr_html_()


def test_catalog_compute_equals_ddf_compute(small_sky_order1_catalog):
    pd.testing.assert_frame_equal(small_sky_order1_catalog.compute(), small_sky_order1_catalog._ddf.compute())


def test_get_catalog_partition_gets_correct_partition(small_sky_order1_catalog):
    for healpix_pixel in small_sky_order1_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = small_sky_order1_catalog.get_partition(hp_order, hp_pixel)
        pixel = HealpixPixel(order=hp_order, pixel=hp_pixel)
        partition_index = small_sky_order1_catalog._ddf_pixel_map[pixel]
        ddf_partition = small_sky_order1_catalog._ddf.partitions[partition_index]
        dd.utils.assert_eq(partition, ddf_partition)


def test_query_no_arguments(small_sky_order1_catalog):
    with pytest.raises(ValueError, match="Provided query expression is not valid"):
        small_sky_order1_catalog.query()


def test_query(small_sky_order1_catalog):
    result_catalog = small_sky_order1_catalog.query("ra > 300 and dec < -50")
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf = expected_ddf[
        (small_sky_order1_catalog._ddf["ra"] > 300) & (small_sky_order1_catalog._ddf["dec"] < -50)
    ]
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())


def test_assign_no_arguments(small_sky_order1_catalog):
    result_catalog = small_sky_order1_catalog.assign()
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), small_sky_order1_catalog.compute())


def test_assign_with_callable(small_sky_order1_catalog):
    kwargs = {"new_column": lambda x: x["ra"] ** 2}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["new_column"] = expected_ddf["ra"] ** 2
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())


def test_assign_with_series(small_sky_order1_catalog):
    # The series is created from the original dataframe because indices must match
    new_series = small_sky_order1_catalog._ddf["ra_error"].map(lambda x: x**2)
    kwargs = {"new_column": new_series}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["new_column"] = new_series
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())
