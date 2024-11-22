from pathlib import Path

import astropy.units as u
import dask.array as da
import dask.dataframe as dd
import hats as hc
import hats.pixel_math.healpix_shim as hp
import matplotlib.pyplot as plt
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes
from hats.inspection.visualize_catalog import get_fov_moc_from_wcs
from hats.io.file_io import read_fits_image
from hats.pixel_math import HealpixPixel, spatial_index_to_healpix
from mocpy import WCS

import lsdb
from lsdb import Catalog
from lsdb.core.search.moc_search import MOCSearch
from lsdb.dask.merge_catalog_functions import filter_by_spatial_index_to_pixel


@pytest.fixture(autouse=True)
def reset_matplotlib():
    yield
    plt.close("all")


def test_catalog_pixels_equals_hc_catalog_pixels(small_sky_order1_catalog, small_sky_order1_hats_catalog):
    assert small_sky_order1_catalog.get_healpix_pixels() == small_sky_order1_hats_catalog.get_healpix_pixels()


def test_catalog_repr_equals_ddf_repr(small_sky_order1_catalog):
    assert repr(small_sky_order1_catalog) == repr(small_sky_order1_catalog._ddf)


def test_catalog_html_repr_equals_ddf_html_repr(small_sky_order1_catalog):
    full_html = small_sky_order1_catalog._repr_html_()
    assert small_sky_order1_catalog.name in full_html
    # this is a _healpix_29 that's in the data
    assert "3170534137668829184" in full_html


def test_catalog_compute_equals_ddf_compute(small_sky_order1_catalog):
    pd.testing.assert_frame_equal(small_sky_order1_catalog.compute(), small_sky_order1_catalog._ddf.compute())


def test_catalog_uses_dask_expressions(small_sky_order1_catalog):
    assert hasattr(small_sky_order1_catalog._ddf, "expr")


def test_get_catalog_partition_gets_correct_partition(small_sky_order1_catalog):
    for healpix_pixel in small_sky_order1_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = small_sky_order1_catalog.get_partition(hp_order, hp_pixel)
        pixel = HealpixPixel(order=hp_order, pixel=hp_pixel)
        partition_index = small_sky_order1_catalog._ddf_pixel_map[pixel]
        ddf_partition = small_sky_order1_catalog._ddf.partitions[partition_index]
        assert isinstance(partition, nd.NestedFrame)
        assert isinstance(partition.compute(), npd.NestedFrame)
        pd.testing.assert_frame_equal(partition.compute(), ddf_partition.compute())


def test_head(small_sky_order1_catalog):
    # By default, head returns 5 rows
    expected_df = small_sky_order1_catalog._ddf.partitions[0].compute()[:5]
    head_df = small_sky_order1_catalog.head()
    assert isinstance(head_df, npd.NestedFrame)
    assert len(head_df) == 5
    pd.testing.assert_frame_equal(expected_df, head_df)
    # But we can also specify the number of rows we desire
    expected_df = small_sky_order1_catalog._ddf.partitions[0].compute()[:10]
    head_df = small_sky_order1_catalog.head(n=10)
    assert len(head_df) == 10
    pd.testing.assert_frame_equal(expected_df, head_df)


def test_head_rows_less_than_requested(small_sky_order1_catalog):
    schema = small_sky_order1_catalog.dtypes
    two_rows = small_sky_order1_catalog._ddf.partitions[0].compute()[:2]
    tiny_df = pd.DataFrame(data=two_rows, columns=schema.index, dtype=schema.to_numpy())
    altered_ndf = nd.NestedFrame.from_pandas(tiny_df, npartitions=1)
    catalog = lsdb.Catalog(altered_ndf, {}, small_sky_order1_catalog.hc_structure)
    # The head only contains two values
    assert len(catalog.head()) == 2


def test_head_first_partition_is_empty(small_sky_order1_catalog):
    # The same catalog but now the first partition is empty
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.to_numpy())
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    altered_ndf = nd.NestedFrame.from_dask_dataframe(dd.concat([empty_ddf, small_sky_order1_catalog._ddf]))
    catalog = lsdb.Catalog(altered_ndf, {}, small_sky_order1_catalog.hc_structure)
    # The first partition is empty
    first_partition_df = catalog._ddf.partitions[0].compute()
    assert len(first_partition_df) == 0
    # We still get values from the second (non-empty) partition
    assert len(catalog.head()) == 5


def test_head_empty_catalog(small_sky_order1_catalog):
    # Create an empty Pandas DataFrame with the same schema
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.to_numpy())
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    empty_catalog = lsdb.Catalog(empty_ddf, {}, small_sky_order1_catalog.hc_structure)
    assert len(empty_catalog.head()) == 0


def test_query(small_sky_order1_catalog):
    expected_ddf = small_sky_order1_catalog._ddf.copy()[
        (small_sky_order1_catalog._ddf["ra"] > 300) & (small_sky_order1_catalog._ddf["dec"] < -50)
    ]
    # Simple query, with no value injection or backticks
    result_catalog = small_sky_order1_catalog.query("ra > 300 and dec < -50")
    assert isinstance(result_catalog._ddf, nd.NestedFrame)
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())
    # Query with value injection
    ra, dec = 300, -50
    result_catalog = small_sky_order1_catalog.query(f"ra > {ra} and dec < {dec}")
    assert isinstance(result_catalog._ddf, nd.NestedFrame)
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())
    # Query with backticks (for invalid Python variables names)
    new_columns = {"ra": "right ascension"}
    expected_ddf = expected_ddf.rename(columns=new_columns)
    small_sky_order1_catalog._ddf = small_sky_order1_catalog._ddf.rename(columns=new_columns)
    result_catalog = small_sky_order1_catalog.query("`right ascension` > 300 and dec < -50")
    assert isinstance(result_catalog._ddf, nd.NestedFrame)
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), expected_ddf.compute())


def test_query_margin(small_sky_xmatch_with_margin):
    expected_ddf = small_sky_xmatch_with_margin._ddf.copy()[
        (small_sky_xmatch_with_margin._ddf["ra"] > 300) & (small_sky_xmatch_with_margin._ddf["dec"] < -50)
    ]
    expected_margin_ddf = small_sky_xmatch_with_margin.margin._ddf.copy()[
        (small_sky_xmatch_with_margin.margin._ddf["ra"] > 300)
        & (small_sky_xmatch_with_margin.margin._ddf["dec"] < -50)
    ]

    result_catalog = small_sky_xmatch_with_margin.query("ra > 300 and dec < -50")
    assert result_catalog.margin is not None
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())
    pd.testing.assert_frame_equal(result_catalog.margin.compute(), expected_margin_ddf.compute())
    assert isinstance(result_catalog.margin._ddf, nd.NestedFrame)


def test_assign_no_arguments(small_sky_order1_catalog):
    result_catalog = small_sky_order1_catalog.assign()
    pd.testing.assert_frame_equal(result_catalog._ddf.compute(), small_sky_order1_catalog._ddf.compute())
    assert isinstance(result_catalog._ddf, nd.NestedFrame)


def test_assign_with_callable(small_sky_order1_catalog):
    kwargs = {"squared_ra_err": lambda x: x["ra_error"] ** 2}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["squared_ra_err"] = expected_ddf["ra_error"] ** 2
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())
    assert isinstance(result_catalog._ddf, nd.NestedFrame)


def test_assign_with_series(small_sky_order1_catalog):
    # The series is created from the original dataframe because indices must match
    squared_ra_err = small_sky_order1_catalog._ddf["ra_error"].map(lambda x: x**2)
    kwargs = {"new_column": squared_ra_err}
    result_catalog = small_sky_order1_catalog.assign(**kwargs)
    expected_ddf = small_sky_order1_catalog._ddf.copy()
    expected_ddf["new_column"] = squared_ra_err
    pd.testing.assert_frame_equal(result_catalog.compute(), expected_ddf.compute())
    assert isinstance(result_catalog._ddf, nd.NestedFrame)


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


def test_save_catalog(small_sky_catalog, tmp_path):
    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.to_hats(base_catalog_path, catalog_name=new_catalog_name)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info
    assert expected_catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_catalog._ddf.compute())


def test_save_catalog_point_map(small_sky_order1_catalog, tmp_path):
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_order1_catalog.to_hats(base_catalog_path, catalog_name=new_catalog_name)

    point_map_path = base_catalog_path / "point_map.fits"
    assert hc.io.file_io.does_file_or_directory_exist(point_map_path)
    histogram = read_fits_image(point_map_path)

    # The histogram and the sky map histogram match
    assert len(small_sky_order1_catalog) == np.sum(histogram)
    expected_histogram = small_sky_order1_catalog.skymap_histogram(lambda df, _: len(df), order=8)
    npt.assert_array_equal(expected_histogram, histogram)


def test_save_catalog_overwrite(small_sky_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    # Saving a catalog to disk when the directory does not yet exist
    small_sky_catalog.to_hats(base_catalog_path)
    # The output directory exists and it has content. Overwrite is
    # set to False and, as such, the operation fails.
    with pytest.raises(ValueError, match="set overwrite to True"):
        small_sky_catalog.to_hats(base_catalog_path)
    # With overwrite it succeeds because the directory is recreated
    small_sky_catalog.to_hats(base_catalog_path, overwrite=True)


def test_save_catalog_when_catalog_is_empty(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"

    # The result of this cone search is known to be empty
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 1)
    assert cone_search_catalog._ddf.npartitions == 1

    non_empty_pixels = []
    for pixel, partition_index in cone_search_catalog._ddf_pixel_map.items():
        if len(cone_search_catalog._ddf.partitions[partition_index]) > 0:
            non_empty_pixels.append(pixel)
    assert len(non_empty_pixels) == 0

    # The catalog is not written to disk
    with pytest.raises(RuntimeError, match="The output catalog is empty"):
        cone_search_catalog.to_hats(base_catalog_path)


def test_save_catalog_with_some_empty_partitions(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"

    # The result of this cone search is known to have one empty partition
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 15 * 3600)
    assert cone_search_catalog._ddf.npartitions == 2

    non_empty_pixels = []
    for pixel, partition_index in cone_search_catalog._ddf_pixel_map.items():
        if len(cone_search_catalog._ddf.partitions[partition_index]) > 0:
            non_empty_pixels.append(pixel)
    assert len(non_empty_pixels) == 1

    cone_search_catalog.to_hats(base_catalog_path)

    # Confirm that we can read the catalog from disk, and that it was
    # written with no empty partitions
    catalog = lsdb.read_hats(base_catalog_path)
    assert catalog._ddf.npartitions == 1
    assert len(catalog._ddf.partitions[0]) > 0
    assert list(catalog._ddf_pixel_map.keys()) == non_empty_pixels


def test_prune_empty_partitions(small_sky_order1_catalog):
    # Perform a query that forces the existence of some empty partitions
    catalog = small_sky_order1_catalog.query("ra > 350 and dec < -50")
    _, non_empty_partitions = catalog._get_non_empty_partitions()
    assert catalog._ddf.npartitions - len(non_empty_partitions) > 0

    with pytest.warns(RuntimeWarning, match="slow"):
        pruned_catalog = catalog.prune_empty_partitions()

    # The empty partitions were removed and the computed content is the same
    _, non_empty_partitions = pruned_catalog._get_non_empty_partitions()
    assert pruned_catalog._ddf.npartitions - len(non_empty_partitions) == 0
    pd.testing.assert_frame_equal(catalog.compute(), pruned_catalog.compute())
    assert isinstance(pruned_catalog._ddf, nd.NestedFrame)
    assert isinstance(pruned_catalog.compute(), npd.NestedFrame)


def test_prune_empty_partitions_with_none_to_remove(small_sky_order1_catalog):
    # The catalog has no empty partitions to be removed
    _, non_empty_partitions = small_sky_order1_catalog._get_non_empty_partitions()
    assert small_sky_order1_catalog._ddf.npartitions == len(non_empty_partitions)

    with pytest.warns(RuntimeWarning, match="slow"):
        pruned_catalog = small_sky_order1_catalog.prune_empty_partitions()

    # The number of partitions and the computed content are the same
    _, non_empty_partitions = pruned_catalog._get_non_empty_partitions()
    assert small_sky_order1_catalog._ddf.npartitions == pruned_catalog._ddf.npartitions
    pd.testing.assert_frame_equal(small_sky_order1_catalog.compute(), pruned_catalog.compute())


def test_prune_empty_partitions_all_are_removed(small_sky_order1_catalog):
    # Perform a query that forces the existence of an empty catalog
    catalog = small_sky_order1_catalog.query("ra > 350 and ra < 350")
    _, non_empty_partitions = catalog._get_non_empty_partitions()
    assert len(non_empty_partitions) == 0

    with pytest.warns(RuntimeWarning, match="slow"):
        pruned_catalog = catalog.prune_empty_partitions()

    # The pruned catalog is also empty
    _, non_empty_partitions = pruned_catalog._get_non_empty_partitions()
    assert len(non_empty_partitions) == 0


def test_skymap_data(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    skymap = small_sky_order1_catalog.skymap_data(func)
    for pixel in skymap.keys():
        partition = small_sky_order1_catalog.get_partition(pixel.order, pixel.pixel)
        expected_value = func(partition, pixel)
        assert skymap[pixel].compute() == expected_value


def test_skymap_data_order(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    order = 3

    skymap = small_sky_order1_catalog.skymap_data(func, order=order)
    for pixel in skymap.keys():
        partition = small_sky_order1_catalog.get_partition(pixel.order, pixel.pixel).compute()
        value = skymap[pixel].compute()
        delta_order = order - pixel.order
        expected_array_length = 1 << 2 * delta_order
        assert len(value) == expected_array_length
        pixels = np.arange(pixel.pixel << (2 * delta_order), (pixel.pixel + 1) << (2 * delta_order))
        for i in range(expected_array_length):
            p = pixels[i]
            expected_value = func(
                filter_by_spatial_index_to_pixel(partition, order, p), HealpixPixel(order, p)
            )
            assert value[i] == expected_value


def test_skymap_data_wrong_order(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    order = 0

    with pytest.raises(ValueError):
        small_sky_order1_catalog.skymap_data(func, order)


def test_skymap_histogram(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    pixel_map = small_sky_order1_catalog.skymap_data(func)
    pixel_map = {pixel: value.compute() for pixel, value in pixel_map.items()}
    max_order = max(pixel_map.keys(), key=lambda x: x.order).order
    img = np.zeros(hp.order2npix(max_order))
    for pixel, value in pixel_map.items():
        dorder = max_order - pixel.order
        start = pixel.pixel * (4**dorder)
        end = (pixel.pixel + 1) * (4**dorder)
        img_order_pixels = np.arange(start, end)
        img[img_order_pixels] = value
    assert (small_sky_order1_catalog.skymap_histogram(func) == img).all()


def test_skymap_histogram_empty(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    expected_img = np.full(12, 1)
    img = small_sky_order1_catalog.cone_search(0, 0, 1).skymap_histogram(func, default_value=1)
    assert (img == expected_img).all()


def test_skymap_histogram_order_default(small_sky_order1_catalog):
    order = 3
    default = -1.0

    def func(df, _):
        return len(df) / hp.order2pixarea(order, degrees=True)

    computed_catalog = small_sky_order1_catalog.compute()
    order_3_pixels = spatial_index_to_healpix(computed_catalog.index.to_numpy(), order)
    pixel_map = computed_catalog.groupby(order_3_pixels).apply(lambda x: func(x, None))

    img = np.full(hp.order2npix(order), default)
    for pixel_num, row in pixel_map.items():
        img[pixel_num] = row
    assert (small_sky_order1_catalog.skymap_histogram(func, order, default) == img).all()


def test_skymap_histogram_null_values_order_default(small_sky_order1_catalog):
    default = -1.0

    def func(df, healpix):
        density = len(df) / hp.order2pixarea(healpix.order, degrees=True)
        return density if healpix.pixel % 2 == 0 else None

    pixels = list(small_sky_order1_catalog._ddf_pixel_map.keys())
    max_order = max(pixels, key=lambda x: x.order).order
    hp_orders = np.vectorize(lambda x: x.order)(pixels)
    hp_pixels = np.vectorize(lambda x: x.pixel)(pixels)

    dorders = max_order - hp_orders
    starts = hp_pixels << (2 * dorders)
    ends = (hp_pixels + 1) << (2 * dorders)

    histogram = small_sky_order1_catalog.skymap_histogram(func, default_value=default)

    for start, end in zip(starts, ends):
        pixels = np.arange(start, end)
        arr = histogram[pixels[pixels % 2 != 0]]
        expected_arr = np.full(arr.shape, fill_value=default)
        assert np.array_equal(expected_arr, arr)


def test_skymap_histogram_null_values_order(small_sky_order1_catalog):
    order = 3
    default = -1.0

    def func(df, healpix):
        density = len(df) / hp.order2pixarea(healpix.order, degrees=True)
        return density if healpix.pixel % 2 == 0 else None

    pixels = list(small_sky_order1_catalog._ddf_pixel_map.keys())
    hp_orders = np.vectorize(lambda x: x.order)(pixels)
    hp_pixels = np.vectorize(lambda x: x.pixel)(pixels)

    dorders = order - hp_orders
    starts = hp_pixels << (2 * dorders)
    ends = (hp_pixels + 1) << (2 * dorders)

    histogram = small_sky_order1_catalog.skymap_histogram(func, order, default)

    for start, end in zip(starts, ends):
        pixels = np.arange(start, end)
        arr = histogram[pixels[pixels % 2 != 0]]
        expected_arr = np.full(arr.shape, fill_value=default)
        assert np.array_equal(expected_arr, arr)


def test_skymap_histogram_order_empty(small_sky_order1_catalog):
    order = 3

    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    catalog = small_sky_order1_catalog.cone_search(0, 0, 1)
    _, non_empty_partitions = catalog._get_non_empty_partitions()
    assert len(non_empty_partitions) == 0

    img = catalog.skymap_histogram(func, order)
    expected_img = np.zeros(hp.order2npix(order))
    assert (img == expected_img).all()


def test_skymap_histogram_order_some_partitions_empty(small_sky_order1_catalog):
    order = 3

    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    catalog = small_sky_order1_catalog.query("ra > 350 and dec < -50")
    _, non_empty_partitions = catalog._get_non_empty_partitions()
    assert 0 < len(non_empty_partitions) < catalog._ddf.npartitions

    img = catalog.skymap_histogram(func, order)

    pixel_map = catalog.skymap_data(func, order)
    pixel_map = {pixel: value.compute() for pixel, value in pixel_map.items()}
    expected_img = np.zeros(hp.order2npix(order))
    for pixel, value in pixel_map.items():
        dorder = order - pixel.order
        start = pixel.pixel * (4**dorder)
        end = (pixel.pixel + 1) * (4**dorder)
        img_order_pixels = np.arange(start, end)
        expected_img[img_order_pixels] = value
    assert (img == expected_img).all()


# pylint: disable=no-member
def test_skymap_plot(small_sky_order1_catalog, mocker):
    mocker.patch("lsdb.catalog.dataset.healpix_dataset.plot_healpix_map")

    def func(df, healpix):
        return len(df) / hp.order2pixarea(healpix.order, degrees=True)

    small_sky_order1_catalog.skymap(func)
    pixel_map = small_sky_order1_catalog.skymap_data(func)
    pixel_map = {pixel: value.compute() for pixel, value in pixel_map.items()}
    max_order = max(pixel_map.keys(), key=lambda x: x.order).order
    img = np.full(hp.order2npix(max_order), 0)
    for pixel, value in pixel_map.items():
        dorder = max_order - pixel.order
        start = pixel.pixel * (4**dorder)
        end = (pixel.pixel + 1) * (4**dorder)
        img_order_pixels = np.arange(start, end)
        img[img_order_pixels] = value
    lsdb.catalog.dataset.healpix_dataset.plot_healpix_map.assert_called_once()
    assert (lsdb.catalog.dataset.healpix_dataset.plot_healpix_map.call_args[0][0] == img).all()


# pylint: disable=no-member
def test_plot_pixels(small_sky_order1_catalog, mocker):
    mocker.patch("hats.catalog.healpix_dataset.healpix_dataset.plot_pixels")
    small_sky_order1_catalog.plot_pixels()

    hc.catalog.healpix_dataset.healpix_dataset.plot_pixels.assert_called_once()
    assert (
        hc.catalog.healpix_dataset.healpix_dataset.plot_pixels.call_args[0][0]
        == small_sky_order1_catalog.hc_structure
    )


# pylint: disable=no-member
def test_plot_coverage(small_sky_order1_catalog, mocker):
    mocker.patch("hats.catalog.healpix_dataset.healpix_dataset.plot_moc")
    small_sky_order1_catalog.plot_coverage()

    hc.catalog.healpix_dataset.healpix_dataset.plot_moc.assert_called_once()
    assert (
        hc.catalog.healpix_dataset.healpix_dataset.plot_moc.call_args[0][0]
        == small_sky_order1_catalog.hc_structure.moc
    )


def test_square_bracket_columns(small_sky_order1_catalog):
    columns = ["ra", "dec", "id"]
    column_subset = small_sky_order1_catalog[columns]
    assert all(column_subset.columns == columns)
    assert isinstance(column_subset, Catalog)
    assert isinstance(column_subset._ddf, nd.NestedFrame)
    assert isinstance(column_subset.compute(), npd.NestedFrame)
    pd.testing.assert_frame_equal(column_subset.compute(), small_sky_order1_catalog.compute()[columns])
    assert np.all(
        column_subset.compute().index.to_numpy() == small_sky_order1_catalog.compute().index.to_numpy()
    )


def test_square_bracket_column(small_sky_order1_catalog):
    column_name = "ra"
    column = small_sky_order1_catalog[column_name]
    pd.testing.assert_series_equal(column.compute(), small_sky_order1_catalog.compute()[column_name])
    assert np.all(column.compute().index.to_numpy() == small_sky_order1_catalog.compute().index.to_numpy())
    assert isinstance(column, dd.Series)


def test_square_bracket_filter(small_sky_order1_catalog):
    filtered_id = small_sky_order1_catalog[small_sky_order1_catalog["id"] > 750]
    assert isinstance(filtered_id, Catalog)
    assert isinstance(filtered_id._ddf, nd.NestedFrame)
    assert isinstance(filtered_id.compute(), npd.NestedFrame)
    ss_computed = small_sky_order1_catalog.compute()
    pd.testing.assert_frame_equal(filtered_id.compute(), ss_computed[ss_computed["id"] > 750])
    assert np.all(
        filtered_id.compute().index.to_numpy() == ss_computed[ss_computed["id"] > 750].index.to_numpy()
    )


def test_map_partitions(small_sky_order1_catalog):
    def add_col(df):
        df["a"] = df["ra"] + 1
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col)
    assert isinstance(mapped, Catalog)
    assert isinstance(mapped._ddf, nd.NestedFrame)
    assert "a" in mapped.columns
    assert mapped.dtypes["a"] == mapped.dtypes["ra"]
    mapcomp = mapped.compute()
    assert isinstance(mapcomp, npd.NestedFrame)
    assert np.all(mapcomp["a"] == mapcomp["ra"] + 1)


def test_map_partitions_include_pixel(small_sky_order1_catalog):
    def add_col(df, pixel):
        df["pix"] = pixel.pixel
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col, include_pixel=True)
    assert isinstance(mapped, Catalog)
    assert "pix" in mapped.columns
    mapcomp = mapped.compute()
    pix_col = hp.radec2pix(1, mapcomp["ra"].to_numpy(), mapcomp["dec"].to_numpy())
    assert np.all(mapcomp["pix"] == pix_col)


def test_map_partitions_specify_meta(small_sky_order1_catalog):
    def add_col(df):
        df["a"] = df["ra"] + 1
        return df

    new_meta = small_sky_order1_catalog.dtypes.to_dict()
    new_meta["a"] = small_sky_order1_catalog.dtypes["ra"]
    mapped = small_sky_order1_catalog.map_partitions(add_col, meta=new_meta)
    assert isinstance(mapped, Catalog)
    assert "a" in mapped.columns
    assert mapped.dtypes["a"] == mapped.dtypes["ra"]
    mapcomp = mapped.compute()
    assert np.all(mapcomp["a"] == mapcomp["ra"] + 1)


def test_map_partitions_non_df(small_sky_order1_catalog):
    def get_col(df):
        return df["ra"] + 1

    with pytest.warns(RuntimeWarning, match="DataFrame"):
        mapped = small_sky_order1_catalog.map_partitions(get_col)

    assert not isinstance(mapped, Catalog)
    assert isinstance(mapped, dd.Series)
    mapcomp = mapped.compute()
    assert np.all(mapcomp == small_sky_order1_catalog.compute()["ra"] + 1)


def test_non_working_empty_raises(small_sky_order1_catalog):
    def add_col(df):
        if len(df) == 0:
            return None
        df["a"] = df["ra"] + 1
        return df

    with pytest.raises(ValueError):
        small_sky_order1_catalog.map_partitions(add_col)


def test_square_bracket_single_partition(small_sky_order1_catalog):
    index = 1
    subset = small_sky_order1_catalog.partitions[index]
    assert isinstance(subset, Catalog)
    assert isinstance(subset._ddf, nd.NestedFrame)
    assert 1 == len(subset._ddf_pixel_map)
    pixel = subset.get_healpix_pixels()[0]
    assert index == small_sky_order1_catalog.get_partition_index(pixel.order, pixel.pixel)
    pd.testing.assert_frame_equal(
        small_sky_order1_catalog._ddf.partitions[index].compute(), subset._ddf.compute()
    )
    assert isinstance(subset.compute(), npd.NestedFrame)


def test_square_bracket_multiple_partitions(small_sky_order1_catalog):
    indices = [0, 1, 2]
    subset = small_sky_order1_catalog.partitions[indices]
    assert isinstance(subset, Catalog)
    assert 3 == len(subset._ddf_pixel_map)
    for pixel, partition_index in subset._ddf_pixel_map.items():
        original_index = small_sky_order1_catalog.get_partition_index(pixel.order, pixel.pixel)
        original_partition = small_sky_order1_catalog._ddf.partitions[original_index]
        subset_partition = subset._ddf.partitions[partition_index]
        pd.testing.assert_frame_equal(original_partition.compute(), subset_partition.compute())


def test_square_bracket_slice_partitions(small_sky_order1_catalog):
    subset = small_sky_order1_catalog.partitions[:2]
    assert isinstance(subset, Catalog)
    assert 2 == len(subset._ddf_pixel_map)
    subset_2 = small_sky_order1_catalog.partitions[0:2]
    assert isinstance(subset, Catalog)
    pd.testing.assert_frame_equal(subset_2.compute(), subset.compute())
    assert subset_2.get_healpix_pixels() == subset.get_healpix_pixels()
    subset_3 = small_sky_order1_catalog.partitions[0:2:1]
    assert subset_3.get_healpix_pixels() == subset.get_healpix_pixels()
    pd.testing.assert_frame_equal(subset_3.compute(), subset.compute())


def test_filtered_catalog_has_undetermined_len(small_sky_order1_catalog, small_sky_order1_id_index_dir):
    """Tests that filtered catalogs have an undetermined number of rows"""
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.query("ra > 300"))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.cone_search(0, -80, 1))
    with pytest.raises(ValueError, match="undetermined"):
        vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
        len(small_sky_order1_catalog.polygon_search(vertices))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.box_search(ra=(280, 300), dec=(0, 30)))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.order_search(max_order=2))
    with pytest.raises(ValueError, match="undetermined"):
        catalog_index = hc.read_hats(small_sky_order1_id_index_dir)
        len(small_sky_order1_catalog.index_search([900], catalog_index))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.pixel_search([(0, 11)]))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.dropna())


def test_joined_catalog_has_undetermined_len(
    small_sky_order1_catalog, small_sky_xmatch_catalog, small_sky_order1_source_with_margin
):
    """Tests that catalogs resulting from joining, merging and crossmatching
    have an undetermined number of rows"""
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.crossmatch(small_sky_xmatch_catalog, radius_arcsec=0.005 * 3600))
    with pytest.raises(ValueError, match="undetermined"):
        len(
            small_sky_order1_catalog.join(
                small_sky_order1_source_with_margin, left_on="id", right_on="object_id"
            )
        )
    with pytest.raises(ValueError, match="undetermined"):
        len(
            small_sky_order1_catalog.join_nested(
                small_sky_order1_source_with_margin,
                left_on="id",
                right_on="object_id",
                nested_column_name="sources",
            )
        )
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.merge_asof(small_sky_xmatch_catalog))


def test_modified_hc_structure_is_a_deep_copy(small_sky_order1_catalog):
    assert small_sky_order1_catalog.hc_structure.pixel_tree is not None
    assert small_sky_order1_catalog.hc_structure.catalog_path is not None
    assert small_sky_order1_catalog.hc_structure.schema is not None
    assert small_sky_order1_catalog.hc_structure.moc is not None
    assert small_sky_order1_catalog.hc_structure.catalog_info.total_rows == 131

    modified_hc_structure = small_sky_order1_catalog._create_modified_hc_structure(total_rows=0)
    modified_hc_structure.pixel_tree = None
    modified_hc_structure.catalog_path = None
    modified_hc_structure.schema = None
    modified_hc_structure.moc = None

    # The original catalog structure is not modified
    assert small_sky_order1_catalog.hc_structure.pixel_tree is not None
    assert small_sky_order1_catalog.hc_structure.catalog_path is not None
    assert small_sky_order1_catalog.hc_structure.schema is not None
    assert small_sky_order1_catalog.hc_structure.moc is not None
    assert small_sky_order1_catalog.hc_structure.catalog_info.total_rows == 131

    # The rows of the new structure are invalidated
    assert modified_hc_structure.catalog_info.total_rows == 0


def test_plot_points(small_sky_order1_catalog, mocker):
    mocker.patch("astropy.visualization.wcsaxes.WCSAxes.scatter")
    _, ax = small_sky_order1_catalog.plot_points()
    comp_cat = small_sky_order1_catalog.compute()
    WCSAxes.scatter.assert_called_once()
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][0], comp_cat["ra"])
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][1], comp_cat["dec"])
    assert WCSAxes.scatter.call_args.kwargs["transform"] == ax.get_transform("icrs")


def test_plot_points_fov(small_sky_order1_catalog, mocker):
    mocker.patch("astropy.visualization.wcsaxes.WCSAxes.scatter")
    fig = plt.figure(figsize=(10, 6))
    center = SkyCoord(350, -80, unit="deg")
    fov = 10 * u.deg
    wcs = WCS(fig=fig, fov=fov, center=center, projection="MOL").w
    wcs_moc = get_fov_moc_from_wcs(wcs)
    _, ax = small_sky_order1_catalog.plot_points(fov=fov, center=center)
    comp_cat = small_sky_order1_catalog.search(MOCSearch(wcs_moc)).compute()
    WCSAxes.scatter.assert_called_once()
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][0], comp_cat["ra"])
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][1], comp_cat["dec"])
    assert WCSAxes.scatter.call_args.kwargs["transform"] == ax.get_transform("icrs")


def test_plot_points_wcs(small_sky_order1_catalog, mocker):
    mocker.patch("astropy.visualization.wcsaxes.WCSAxes.scatter")
    fig = plt.figure(figsize=(10, 6))
    center = SkyCoord(350, -80, unit="deg")
    fov = 10 * u.deg
    wcs = WCS(fig=fig, fov=fov, center=center).w
    wcs_moc = get_fov_moc_from_wcs(wcs)
    _, ax = small_sky_order1_catalog.plot_points(wcs=wcs)
    comp_cat = small_sky_order1_catalog.search(MOCSearch(wcs_moc)).compute()
    WCSAxes.scatter.assert_called_once()
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][0], comp_cat["ra"])
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][1], comp_cat["dec"])
    assert WCSAxes.scatter.call_args.kwargs["transform"] == ax.get_transform("icrs")


def test_plot_points_colorcol(small_sky_order1_catalog, mocker):
    mocker.patch("astropy.visualization.wcsaxes.WCSAxes.scatter")
    mocker.patch("matplotlib.pyplot.colorbar")
    _, ax = small_sky_order1_catalog.plot_points(color_col="id")
    comp_cat = small_sky_order1_catalog.compute()
    WCSAxes.scatter.assert_called_once()
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][0], comp_cat["ra"])
    npt.assert_array_equal(WCSAxes.scatter.call_args[0][1], comp_cat["dec"])
    npt.assert_array_equal(WCSAxes.scatter.call_args.kwargs["c"], comp_cat["id"])
    assert WCSAxes.scatter.call_args.kwargs["transform"] == ax.get_transform("icrs")
    plt.colorbar.assert_called_once()
