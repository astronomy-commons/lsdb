# pylint: disable=too-many-lines
from pathlib import Path

import astropy.units as u
import dask.dataframe as dd
import hats as hc
import hats.pixel_math.healpix_shim as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import nested_pandas as npd
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes
from hats.inspection.visualize_catalog import get_fov_moc_from_wcs
from hats.pixel_math import HealpixPixel
from mocpy import WCS
from nested_pandas.datasets import generate_data

import lsdb
import lsdb.nested as nd
from lsdb import Catalog, MarginCatalog
from lsdb.core.search.region_search import MOCSearch

mpl.use("Agg")


@pytest.fixture(autouse=True)
def reset_matplotlib():
    yield
    plt.close("all")


def test_catalog_pixels_equals_hc_catalog_pixels(small_sky_order1_catalog, small_sky_order1_hats_catalog):
    assert small_sky_order1_catalog.get_healpix_pixels() == small_sky_order1_hats_catalog.get_healpix_pixels()


def test_catalog_repr_equals_ddf_repr(small_sky_order1_catalog):
    assert repr(small_sky_order1_catalog) == repr(small_sky_order1_catalog._ddf)


def test_catalog_html_repr(small_sky_order1_catalog):
    full_html = small_sky_order1_catalog._repr_html_()
    assert small_sky_order1_catalog.name in full_html
    assert str(small_sky_order1_catalog.get_ordered_healpix_pixels()[0]) in full_html
    assert str(small_sky_order1_catalog.get_ordered_healpix_pixels()[-1]) in full_html
    assert "available columns in the catalog have been loaded <strong>lazily</strong>" in full_html


def test_catalog_html_repr_empty(small_sky_order1_catalog):
    pixel_search = lsdb.PixelSearch.from_radec(80.0, 33.0)
    cat = small_sky_order1_catalog.search(pixel_search)
    full_html = cat._repr_html_()
    assert cat.name in full_html
    assert "Empty Catalog" in full_html
    assert "npartitions=0" in full_html
    assert "available columns in the catalog have been loaded <strong>lazily</strong>" in full_html


def test_catalog_compute_equals_ddf_compute(small_sky_order1_catalog):
    pd.testing.assert_frame_equal(small_sky_order1_catalog.compute(), small_sky_order1_catalog._ddf.compute())


def test_catalog_uses_dask_expressions(small_sky_order1_catalog):
    assert hasattr(small_sky_order1_catalog._ddf, "expr")


def test_catalog_iloc_raises_error(small_sky_order1_catalog):
    with pytest.raises(NotImplementedError, match="computing the entire catalog"):
        _ = small_sky_order1_catalog.iloc[0]


def test_catalog_loc_raises_error(small_sky_order1_catalog):
    with pytest.raises(NotImplementedError, match="id_search"):
        _ = small_sky_order1_catalog.loc[707]


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


def test_npartitions_property(small_sky_order1_catalog):
    underlying_count = len(small_sky_order1_catalog.get_healpix_pixels())
    assert underlying_count == small_sky_order1_catalog.npartitions


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


def test_tail(small_sky_order1_catalog):
    # By default, tail returns 5 rows
    expected_df = small_sky_order1_catalog._ddf.partitions[0].compute()[-5:]
    tail_df = small_sky_order1_catalog.tail()
    assert isinstance(tail_df, npd.NestedFrame)
    assert len(tail_df) == 5
    pd.testing.assert_frame_equal(expected_df, tail_df)
    # But we can also specify the number of rows we desire
    expected_df = small_sky_order1_catalog._ddf.partitions[0].compute()[-10:]
    tail_df = small_sky_order1_catalog.tail(n=10)
    assert len(tail_df) == 10
    pd.testing.assert_frame_equal(expected_df, tail_df)


def test_tail_rows_less_than_requested(small_sky_order1_catalog):
    schema = small_sky_order1_catalog.dtypes
    two_rows = small_sky_order1_catalog._ddf.partitions[0].compute()[-2:]
    tiny_df = pd.DataFrame(data=two_rows, columns=schema.index, dtype=schema.to_numpy())
    altered_ndf = nd.NestedFrame.from_pandas(tiny_df, npartitions=1)
    catalog = lsdb.Catalog(altered_ndf, {}, small_sky_order1_catalog.hc_structure)
    # The tail only contains two values
    assert len(catalog.tail()) == 2


def test_tail_first_partition_is_empty(small_sky_order1_catalog):
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
    assert len(catalog.tail()) == 5


def test_tail_empty_catalog(small_sky_order1_catalog):
    # Create an empty Pandas DataFrame with the same schema
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.to_numpy())
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    empty_catalog = lsdb.Catalog(empty_ddf, {}, small_sky_order1_catalog.hc_structure)
    assert len(empty_catalog.tail()) == 0


def test_random_sample(small_sky_order1_catalog):
    # By default, random_sample returns 5 rows
    rsample_df = small_sky_order1_catalog.random_sample(seed=0)
    assert isinstance(rsample_df, npd.NestedFrame)
    assert len(rsample_df) == 5
    # But we can also specify the number of rows we desire
    rsample_df = small_sky_order1_catalog.random_sample(n=10, seed=0)
    assert len(rsample_df) == 10


def test_random_sample_changed_catalog(small_sky_order1_catalog):
    # Choose a small portion of this catalog so that random_sample
    # can't rely on the precalculated pixel statistics.

    # The result of this cone search is known to have one empty partition
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 15 * 3600)
    rsample_df = cone_search_catalog.random_sample(n=3, seed=0)
    assert isinstance(rsample_df, npd.NestedFrame)
    assert len(rsample_df) == 3


def test_random_sample_no_return(small_sky_order1_catalog):
    # Test what happens when no elements are returned.
    rsample_df = small_sky_order1_catalog.random_sample(n=0, seed=0)
    assert isinstance(rsample_df, npd.NestedFrame)
    assert len(rsample_df) == 0


def test_random_sample_empty_catalog(small_sky_order1_catalog):
    # Create an empty Pandas DataFrame with the same schema
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.to_numpy())
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    empty_catalog = lsdb.Catalog(empty_ddf, {}, small_sky_order1_catalog.hc_structure)
    assert len(empty_catalog.random_sample(seed=0)) == 0


def test_sample(small_sky_order1_catalog):
    # By default, sample returns 5 rows
    sample_df = small_sky_order1_catalog.sample(partition_id=0, seed=0)
    assert isinstance(sample_df, npd.NestedFrame)
    assert len(sample_df) == 5
    # But we can also specify the number of rows we desire
    sample_df = small_sky_order1_catalog.sample(partition_id=0, n=10, seed=0)
    assert len(sample_df) == 10
    # Verify that partition_id is verified to be in range
    with pytest.raises(IndexError):
        small_sky_order1_catalog.sample(partition_id=-1, n=10, seed=0)
    with pytest.raises(IndexError):
        npartitions = len(small_sky_order1_catalog.get_healpix_pixels())
        small_sky_order1_catalog.sample(partition_id=npartitions, n=10)


def test_sample_empty_catalog(small_sky_order1_catalog):
    # Create an empty Pandas DataFrame with the same schema
    schema = small_sky_order1_catalog.dtypes
    empty_df = pd.DataFrame(columns=schema.index, dtype=schema.to_numpy())
    empty_ddf = dd.from_pandas(empty_df, npartitions=1)
    empty_catalog = lsdb.Catalog(empty_ddf, {}, small_sky_order1_catalog.hc_structure)
    assert len(empty_catalog.sample(partition_id=0, seed=0)) == 0


def test_query(small_sky_order1_catalog, helpers):
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
    helpers.assert_schema_correct(result_catalog)
    assert result_catalog.hc_structure.catalog_path is not None


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


def test_rename_with_callable(small_sky_xmatch_with_margin):
    uppercase_catalog = small_sky_xmatch_with_margin.rename(columns=str.upper)
    assert len(small_sky_xmatch_with_margin.columns) == len(
        uppercase_catalog.columns == len(uppercase_catalog.margin.columns)
    )
    for i, col in enumerate(small_sky_xmatch_with_margin.columns):
        colname = col.upper()
        assert uppercase_catalog.columns[i] == colname
        assert uppercase_catalog.margin.columns[i] == colname

    lowercase_catalog = uppercase_catalog.rename(columns=str.lower)
    assert len(small_sky_xmatch_with_margin.columns) == len(
        lowercase_catalog.columns == len(lowercase_catalog.margin.columns)
    )
    for i, col in enumerate(small_sky_xmatch_with_margin.columns):
        colname = col.lower()
        assert lowercase_catalog.columns[i] == colname
        assert lowercase_catalog.margin.columns[i] == colname


def test_rename_with_dict(small_sky_xmatch_with_margin):
    rename_map = {}
    for i, col in enumerate(small_sky_xmatch_with_margin.columns):
        rename_map[col] = f"{col}_{i}"
    renamed_catalog = small_sky_xmatch_with_margin.rename(columns=rename_map)

    assert (
        len(small_sky_xmatch_with_margin.columns)
        == len(renamed_catalog.columns)
        == len(renamed_catalog.margin.columns)
    )
    for i, col in enumerate(small_sky_xmatch_with_margin.columns):
        assert renamed_catalog.columns[i] == f"{col}_{i}"
        assert renamed_catalog.margin.columns[i] == f"{col}_{i}"


def test_read_hats(small_sky_catalog, tmp_path):
    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.to_hats(base_catalog_path, catalog_name=new_catalog_name)

    # Using .read_hats here vs .open_catalog in order to exercise code coverage for .read_hats
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.schema.pandas_metadata is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_catalog._ddf.compute())


def test_advise_unloaded_columns(small_sky_order1_default_cols_catalog):
    cat = small_sky_order1_default_cols_catalog
    with pytest.raises(ValueError, match="Column `dec_error` is in the catalog but was not loaded."):
        _ = cat["dec_error"]
    with pytest.raises(
        ValueError, match="Columns `ra_error`, `dec_error` are in the catalog but were not loaded."
    ):
        _ = cat[["ra_error", "dec_error"]]


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


def test_aggregate_column_statistics(small_sky_order1_catalog):
    def assert_column_stat_as_floats(
        result_frame, column_name, min_value=None, max_value=None, row_count=None
    ):
        assert column_name in result_frame.index
        data_stats = result_frame.loc[column_name]
        assert float(data_stats["min_value"]) >= min_value
        assert float(data_stats["max_value"]) <= max_value
        assert int(data_stats["null_count"]) == 0
        assert int(data_stats["row_count"]) == row_count

    result_frame = small_sky_order1_catalog.aggregate_column_statistics()
    assert len(result_frame) == 5
    assert_column_stat_as_floats(result_frame, "dec", min_value=-69.5, max_value=-25.5, row_count=131)

    result_frame = small_sky_order1_catalog.aggregate_column_statistics(exclude_hats_columns=False)
    assert_column_stat_as_floats(
        result_frame, "_healpix_29", min_value=1, max_value=3_458_764_513_820_540_928, row_count=131
    )
    assert len(result_frame) == 6

    result_frame = small_sky_order1_catalog.aggregate_column_statistics(include_columns=["ra", "dec"])
    assert len(result_frame) == 2

    filtered_catalog = small_sky_order1_catalog.cone_search(315, -66.443, 0.1, fine=False)
    result_frame = filtered_catalog.aggregate_column_statistics()
    assert len(result_frame) == 5
    assert_column_stat_as_floats(result_frame, "dec", min_value=-69.5, max_value=-47.5, row_count=42)


def test_per_pixel_statistics(small_sky_order1_catalog):
    result_frame = small_sky_order1_catalog.per_pixel_statistics()
    assert result_frame.shape == (4, 20)

    result_frame = small_sky_order1_catalog.per_pixel_statistics(exclude_hats_columns=False)
    assert result_frame.shape == (4, 24)

    result_frame = small_sky_order1_catalog.per_pixel_statistics(include_columns=["ra", "dec"])
    assert result_frame.shape == (4, 8)

    filtered_catalog = small_sky_order1_catalog.cone_search(315, -66.443, 0.1, fine=False)
    result_frame = filtered_catalog.per_pixel_statistics(
        include_stats=["row_count"], include_columns=["ra", "dec"]
    )
    assert result_frame.shape == (1, 2)


def test_square_bracket_columns(small_sky_order1_catalog, helpers):
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
    helpers.assert_schema_correct(column_subset)


def test_square_bracket_columns_default_columns(small_sky_order1_default_cols_catalog, helpers):
    columns = ["ra", "dec"]
    column_subset = small_sky_order1_default_cols_catalog[columns]
    assert all(column_subset.columns == columns)
    assert isinstance(column_subset, Catalog)
    assert isinstance(column_subset._ddf, nd.NestedFrame)
    assert isinstance(column_subset.compute(), npd.NestedFrame)
    pd.testing.assert_frame_equal(
        column_subset.compute(), small_sky_order1_default_cols_catalog.compute()[columns]
    )
    assert np.all(
        column_subset.compute().index.to_numpy()
        == small_sky_order1_default_cols_catalog.compute().index.to_numpy()
    )
    helpers.assert_schema_correct(column_subset)
    helpers.assert_default_columns_in_columns(column_subset)


def test_square_bracket_column(small_sky_order1_catalog):
    column_name = "ra"
    column = small_sky_order1_catalog[column_name]
    pd.testing.assert_series_equal(column.compute(), small_sky_order1_catalog.compute()[column_name])
    assert np.all(column.compute().index.to_numpy() == small_sky_order1_catalog.compute().index.to_numpy())
    assert isinstance(column, dd.Series)


def test_square_bracket_filter(small_sky_order1_catalog, helpers):
    filtered_id = small_sky_order1_catalog[small_sky_order1_catalog["id"] > 750]
    assert isinstance(filtered_id, Catalog)
    assert isinstance(filtered_id._ddf, nd.NestedFrame)
    assert isinstance(filtered_id.compute(), npd.NestedFrame)
    ss_computed = small_sky_order1_catalog.compute()
    pd.testing.assert_frame_equal(filtered_id.compute(), ss_computed[ss_computed["id"] > 750])
    assert np.all(
        filtered_id.compute().index.to_numpy() == ss_computed[ss_computed["id"] > 750].index.to_numpy()
    )
    helpers.assert_schema_correct(filtered_id)


def test_map_partitions(small_sky_order1_catalog):
    def add_col(df, new_col_name, *, increment_value):
        df[new_col_name] = df["ra"] + increment_value
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col, "a", increment_value=1)
    assert isinstance(mapped, Catalog)
    assert isinstance(mapped._ddf, nd.NestedFrame)
    assert "a" in mapped.columns
    assert mapped.dtypes["a"] == mapped.dtypes["ra"]
    mapcomp = mapped.compute()
    assert isinstance(mapcomp, npd.NestedFrame)
    assert np.all(mapcomp["a"] == mapcomp["ra"] + 1)
    assert mapped.hc_structure.catalog_path is not None


def test_map_partitions_include_pixel(small_sky_order1_catalog):
    def add_col(df, pixel, new_col_name, *, increment_value):
        df[new_col_name] = pixel.pixel + increment_value
        return df

    mapped = small_sky_order1_catalog.map_partitions(add_col, "pix", increment_value=1, include_pixel=True)
    assert isinstance(mapped, Catalog)
    assert "pix" in mapped.columns
    mapcomp = mapped.compute()
    pix_col = hp.radec2pix(1, mapcomp["ra"].to_numpy(), mapcomp["dec"].to_numpy()) + 1
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


def test_map_partitions_updates_margin(small_sky_order1_source_with_margin):
    def add_col(df, new_col_name, *, increment_value):
        df[new_col_name] = df["source_ra"] + increment_value
        return df

    mapped = small_sky_order1_source_with_margin.map_partitions(add_col, "a", increment_value=1)
    assert isinstance(mapped, Catalog)
    assert isinstance(mapped.margin, MarginCatalog)
    assert "a" in mapped.margin.columns
    assert mapped.margin.dtypes["a"] == mapped.margin.dtypes["source_ra"]
    mapped_margin_comp = mapped.margin.compute()
    assert isinstance(mapped_margin_comp, npd.NestedFrame)
    assert np.all(mapped_margin_comp["a"] == mapped_margin_comp["source_ra"] + 1)
    assert mapped.margin.hc_structure.catalog_path is not None


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
        len(small_sky_order1_catalog.box_search(ra=(280, 300), dec=(0, 30)))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.order_search(max_order=2))
    with pytest.raises(ValueError, match="undetermined"):
        catalog_index = hc.read_hats(small_sky_order1_id_index_dir)
        len(small_sky_order1_catalog.id_search(values={"id": 900}, index_catalogs={"id": catalog_index}))
    with pytest.raises(ValueError, match="undetermined"):
        len(small_sky_order1_catalog.pixel_search([(0, 11)]))


@pytest.mark.sphgeom
def test_filtered_catalog_has_undetermined_len_polygon(small_sky_order1_catalog):
    """Tests that filtered catalogs have an undetermined number of rows"""
    with pytest.raises(ValueError, match="undetermined"):
        vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
        len(small_sky_order1_catalog.polygon_search(vertices))


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


def test_all_columns(small_sky_order1_default_cols_catalog, small_sky_order1_catalog):
    assert len(small_sky_order1_default_cols_catalog.columns) < len(
        small_sky_order1_default_cols_catalog.all_columns
    )
    assert np.all(small_sky_order1_default_cols_catalog.all_columns == small_sky_order1_catalog.columns)


def test_original_schema(small_sky_order1_catalog):
    assert small_sky_order1_catalog.original_schema is not None
    filtered_cat = small_sky_order1_catalog[["ra", "dec"]]
    assert filtered_cat.original_schema == small_sky_order1_catalog.original_schema


def test_all_columns_after_query(small_sky_order1_catalog):
    filtered_cat = small_sky_order1_catalog.query("ra > 10")
    assert filtered_cat.all_columns == small_sky_order1_catalog.all_columns


def test_all_columns_after_column_select(small_sky_order1_catalog):
    filtered_cat = small_sky_order1_catalog[["ra", "dec"]]
    assert filtered_cat.all_columns == small_sky_order1_catalog.all_columns


def test_all_columns_after_filter(small_sky_order1_catalog):
    filtered_cat = small_sky_order1_catalog.cone_search(10, 10, 10)
    assert filtered_cat.all_columns == small_sky_order1_catalog.all_columns


def test_map_partitions_error_messages():
    # Create a dummy catalog with few partitions
    nf = generate_data(5, 5, seed=0)
    nf.loc[2, "a"] = 0.0  # Introduce a zero to trigger an error in the user function

    def divme(df, _=None):
        if (df["a"] == 0.0).any():
            raise ValueError("Not so fast")
        return 1 / df["a"]

    # Force every row into a separate partition
    nfc = lsdb.from_dataframe(nf, ra_column="a", dec_column="b", partition_size=1)

    with pytest.raises(RuntimeError, match=r"function divme to partition 3: Not so fast"):
        nfc.map_partitions(divme, include_pixel=False).compute()

    with pytest.raises(
        RuntimeError, match=r"function divme to partition 3, pixel Order: 7, Pixel: 77836: Not so fast"
    ):
        nfc.map_partitions(divme, include_pixel=True).compute()
