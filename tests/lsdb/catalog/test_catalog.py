import os
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import healpy as hp
import numpy as np
import pandas as pd
import pytest
from hipscat.pixel_math import HealpixPixel, hipscat_id_to_healpix

import lsdb
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import filter_by_hipscat_index_to_pixel


def test_catalog_pixels_equals_hc_catalog_pixels(small_sky_order1_catalog, small_sky_order1_hipscat_catalog):
    assert (
        small_sky_order1_catalog.get_healpix_pixels() == small_sky_order1_hipscat_catalog.get_healpix_pixels()
    )


def test_catalog_repr_equals_ddf_repr(small_sky_order1_catalog):
    assert repr(small_sky_order1_catalog) == repr(small_sky_order1_catalog._ddf)


def test_catalog_html_repr_equals_ddf_html_repr(small_sky_order1_catalog):
    full_html = small_sky_order1_catalog._repr_html_()
    assert small_sky_order1_catalog.name in full_html
    # this is a _hipscat_index that's in the data
    assert "12682136550675316736" in full_html


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


def test_head(small_sky_order1_catalog):
    # By default, head returns 5 rows
    expected_df = small_sky_order1_catalog._ddf.partitions[0].compute()[:5]
    head_df = small_sky_order1_catalog.head()
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
    tiny_df = pd.DataFrame(data=two_rows, columns=schema.index, dtype=schema.values)
    altered_ddf = dd.io.from_pandas(tiny_df, npartitions=1)
    catalog = lsdb.Catalog(altered_ddf, {}, small_sky_order1_catalog.hc_structure)
    # The head only contains two values
    assert len(catalog.head()) == 2


def test_head_first_partition_is_empty(small_sky_order1_catalog):
    # The same catalog but now the first partition is empty
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.values)
    empty_ddf = dd.io.from_pandas(empty_df, npartitions=1)
    altered_ddf = dd.multi.concat([empty_ddf, small_sky_order1_catalog._ddf])
    catalog = lsdb.Catalog(altered_ddf, {}, small_sky_order1_catalog.hc_structure)
    # The first partition is empty
    first_partition_df = catalog._ddf.partitions[0].compute()
    assert len(first_partition_df) == 0
    # We still get values from the second (non-empty) partition
    assert len(catalog.head()) == 5


def test_head_empty_catalog(small_sky_order1_catalog):
    # Create an empty Pandas DataFrame with the same schema
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.values)
    empty_ddf = dd.io.from_pandas(empty_df, npartitions=1)
    empty_catalog = lsdb.Catalog(empty_ddf, {}, small_sky_order1_catalog.hc_structure)
    assert len(empty_catalog.head()) == 0


def test_query(small_sky_order1_catalog):
    expected_ddf = small_sky_order1_catalog._ddf.copy()[
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
    with pytest.raises(ValueError):
        small_sky_order1_catalog.query(None)


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


def test_save_catalog(small_sky_catalog, tmp_path):
    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.to_hipscat(base_catalog_path, catalog_name=new_catalog_name)
    expected_catalog = lsdb.read_hipscat(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info
    assert expected_catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_catalog._ddf.compute())


def test_save_catalog_overwrite(small_sky_catalog, tmp_path):
    base_catalog_path = os.path.join(tmp_path, "small_sky")
    small_sky_catalog.to_hipscat(base_catalog_path)
    with pytest.raises(FileExistsError):
        small_sky_catalog.to_hipscat(base_catalog_path)
    small_sky_catalog.to_hipscat(base_catalog_path, overwrite=True)


def test_save_catalog_when_catalog_is_empty(small_sky_order1_catalog, tmp_path):
    base_catalog_path = os.path.join(tmp_path, "small_sky")

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
        cone_search_catalog.to_hipscat(base_catalog_path)


def test_save_catalog_with_some_empty_partitions(small_sky_order1_catalog, tmp_path):
    base_catalog_path = os.path.join(tmp_path, "small_sky")

    # The result of this cone search is known to have one empty partition
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 15 * 3600)
    assert cone_search_catalog._ddf.npartitions == 2

    non_empty_pixels = []
    for pixel, partition_index in cone_search_catalog._ddf_pixel_map.items():
        if len(cone_search_catalog._ddf.partitions[partition_index]) > 0:
            non_empty_pixels.append(pixel)
    assert len(non_empty_pixels) == 1

    cone_search_catalog.to_hipscat(base_catalog_path)

    # Confirm that we can read the catalog from disk, and that it was
    # written with no empty partitions
    catalog = lsdb.read_hipscat(base_catalog_path)
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
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

    skymap = small_sky_order1_catalog.skymap_data(func)
    for pixel in skymap.keys():
        partition = small_sky_order1_catalog.get_partition(pixel.order, pixel.pixel)
        expected_value = func(partition, pixel)
        assert skymap[pixel].compute() == expected_value


def test_skymap_data_order(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

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
                filter_by_hipscat_index_to_pixel(partition, order, p), HealpixPixel(order, p)
            )
            assert value[i] == expected_value


def test_skymap_data_wrong_order(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

    order = 0

    with pytest.raises(ValueError):
        small_sky_order1_catalog.skymap_data(func, order)


def test_skymap_histogram(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

    pixel_map = small_sky_order1_catalog.skymap_data(func)
    pixel_map = {pixel: value.compute() for pixel, value in pixel_map.items()}
    max_order = max(pixel_map.keys(), key=lambda x: x.order).order
    img = np.zeros(hp.nside2npix(hp.order2nside(max_order)))
    for pixel, value in pixel_map.items():
        dorder = max_order - pixel.order
        start = pixel.pixel * (4**dorder)
        end = (pixel.pixel + 1) * (4**dorder)
        img_order_pixels = np.arange(start, end)
        img[img_order_pixels] = value
    assert (small_sky_order1_catalog.skymap_histogram(func) == img).all()


def test_skymap_histogram_empty(small_sky_order1_catalog):
    def func(df, healpix):
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

    expected_img = np.full(12, 1)
    img = small_sky_order1_catalog.cone_search(0, 0, 1).skymap_histogram(func, default_value=1)
    assert (img == expected_img).all()


def test_skymap_histogram_order_default(small_sky_order1_catalog):
    order = 3
    default = -1.0

    def func(df, _):
        return len(df) / hp.nside2pixarea(hp.order2nside(order), degrees=True)

    computed_catalog = small_sky_order1_catalog.compute()
    order_3_pixels = hipscat_id_to_healpix(computed_catalog.index.values, order)
    pixel_map = computed_catalog.groupby(order_3_pixels).apply(lambda x: func(x, None))

    img = np.full(hp.nside2npix(hp.order2nside(order)), default)
    for pixel_num, row in pixel_map.items():
        img[pixel_num] = row
    assert (small_sky_order1_catalog.skymap_histogram(func, order, default) == img).all()


# pylint: disable=no-member
def test_skymap_plot(small_sky_order1_catalog, mocker):
    mocker.patch("healpy.mollview")

    def func(df, healpix):
        return len(df) / hp.nside2pixarea(hp.order2nside(healpix.order), degrees=True)

    small_sky_order1_catalog.skymap(func)
    pixel_map = small_sky_order1_catalog.skymap_data(func)
    pixel_map = {pixel: value.compute() for pixel, value in pixel_map.items()}
    max_order = max(pixel_map.keys(), key=lambda x: x.order).order
    img = np.full(hp.nside2npix(hp.order2nside(max_order)), hp.pixelfunc.UNSEEN)
    for pixel, value in pixel_map.items():
        dorder = max_order - pixel.order
        start = pixel.pixel * (4**dorder)
        end = (pixel.pixel + 1) * (4**dorder)
        img_order_pixels = np.arange(start, end)
        img[img_order_pixels] = value
    hp.mollview.assert_called_once()
    assert (hp.mollview.call_args[0][0] == img).all()


# pylint: disable=no-member
def test_plot_pixels(small_sky_order1_catalog, mocker):
    mocker.patch("healpy.mollview")

    small_sky_order1_catalog.plot_pixels()

    # Everything will be empty, except the four pixels at order 1.
    img = np.full(48, hp.pixelfunc.UNSEEN)
    img[[44, 45, 46, 47]] = 1

    hp.mollview.assert_called_once()
    assert (hp.mollview.call_args[0][0] == img).all()


def test_square_bracket_columns(small_sky_order1_catalog):
    columns = ["ra", "dec", "id"]
    column_subset = small_sky_order1_catalog[columns]
    assert all(column_subset.columns == columns)
    assert isinstance(column_subset, Catalog)
    pd.testing.assert_frame_equal(column_subset.compute(), small_sky_order1_catalog.compute()[columns])
    assert np.all(column_subset.compute().index.values == small_sky_order1_catalog.compute().index.values)


def test_square_bracket_column(small_sky_order1_catalog):
    column_name = "ra"
    column = small_sky_order1_catalog[column_name]
    pd.testing.assert_series_equal(column.compute(), small_sky_order1_catalog.compute()[column_name])
    assert np.all(column.compute().index.values == small_sky_order1_catalog.compute().index.values)
    assert isinstance(column, dd.core.Series)


def test_square_bracket_filter(small_sky_order1_catalog):
    filtered_id = small_sky_order1_catalog[small_sky_order1_catalog["id"] > 750]
    assert isinstance(filtered_id, Catalog)
    ss_computed = small_sky_order1_catalog.compute()
    pd.testing.assert_frame_equal(filtered_id.compute(), ss_computed[ss_computed["id"] > 750])
    assert np.all(filtered_id.compute().index.values == ss_computed[ss_computed["id"] > 750].index.values)


def test_map_partitions(small_sky_order1_catalog):
    def add_col(df):
        df["a"] = df["ra"] + 1
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col)
    assert isinstance(mapped, Catalog)
    assert "a" in mapped.columns
    assert mapped.dtypes["a"] == mapped.dtypes["ra"]
    mapcomp = mapped.compute()
    assert np.all(mapcomp["a"] == mapcomp["ra"] + 1)


def test_map_partitions_include_pixel(small_sky_order1_catalog):
    def add_col(df, pixel):
        df["pix"] = pixel.pixel
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col, include_pixel=True)
    assert isinstance(mapped, Catalog)
    assert "pix" in mapped.columns
    mapcomp = mapped.compute()
    pix_col = hp.ang2pix(
        hp.order2nside(1), mapcomp["ra"].to_numpy(), mapcomp["dec"].to_numpy(), lonlat=True, nest=True
    )
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


def test_non_working_empty_raises(small_sky_order1_catalog):
    def add_col(df):
        if len(df) == 0:
            return None
        df["a"] = df["ra"] + 1
        return df

    with pytest.raises(ValueError):
        small_sky_order1_catalog.map_partitions(add_col)
