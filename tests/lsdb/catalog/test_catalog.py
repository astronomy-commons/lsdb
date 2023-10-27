import dask.array as da
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


def test_query(small_sky_order1_catalog):
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf = expected_ddf[
        (small_sky_order1_catalog._ddf["ra"] > 300) & (small_sky_order1_catalog._ddf["dec"] < -50)
    ]
    # Simple query, with no value injection or backticks
    result_catalog = small_sky_order1_catalog.query("ra > 300 and dec < -50")
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())
    # Query with value injection
    ra, dec = 300, -50
    result_catalog = small_sky_order1_catalog.query(f"ra > {ra} and dec < {dec}")
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())
    # Query with backticks (for invalid Python variables names)
    new_columns = {"ra": "right ascension"}
    expected_ddf = expected_ddf.rename(columns=new_columns)
    small_sky_order1_catalog._ddf = small_sky_order1_catalog._ddf.rename(columns=new_columns)
    result_catalog = small_sky_order1_catalog.query("`right ascension` > 300 and dec < -50")
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())


def test_query_no_arguments(small_sky_order1_catalog):
    with pytest.raises(ValueError, match="Provided query expression is not valid"):
        small_sky_order1_catalog.query()


def test_assign_no_arguments(small_sky_order1_catalog):
    result_catalog = small_sky_order1_catalog.assign()
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), small_sky_order1_catalog._ddf.compute())


def test_assign_with_callable(small_sky_order1_catalog):
    kwargs = {"squared_ra_err": lambda x: x["ra_error"] ** 2}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["squared_ra_err"] = expected_ddf["ra_error"] ** 2
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())


def test_assign_with_series(small_sky_order1_catalog):
    # The series is created from the original dataframe because indices must match
    squared_ra_err = small_sky_order1_catalog._ddf["ra_error"].map(lambda x: x**2)
    kwargs = {"new_column": squared_ra_err}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["new_column"] = squared_ra_err
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())


def test_assign_with_multiple_columns(small_sky_order1_catalog):
    # These series are created from the original dataframe because indices must match
    squared_ra_err = small_sky_order1_catalog._ddf["ra_error"].map(lambda x: x**2)
    squared_dec_err = small_sky_order1_catalog._ddf["dec_error"].map(lambda x: x**2)
    kwargs = {
        "squared_ra_err": squared_ra_err,
        "squared_dec_err": squared_dec_err,
    }
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["squared_ra_err"] = squared_ra_err
    expected_ddf["squared_dec_err"] = squared_dec_err
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())


def test_assign_with_invalid_arguments(small_sky_order1_catalog):
    with pytest.raises(TypeError, match="Column assignment doesn't support type"):
        small_sky_order1_catalog.assign(new_column=[1, 2, 3])
    with pytest.raises(ValueError, match="Array assignment only supports 1-D arrays"):
        small_sky_order1_catalog.assign(new_column=da.ones((10, 10)))
    with pytest.raises(ValueError, match="Number of partitions do not match"):
        chunks = small_sky_order1_catalog._ddf.npartitions + 1
        array = da.random.random(size=10, chunks=chunks)
        small_sky_order1_catalog.assign(new_column=array)
