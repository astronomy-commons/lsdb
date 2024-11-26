from pathlib import Path

import hats as hc
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, compute_spatial_index
from pandas.core.dtypes.base import ExtensionDtype

import lsdb
from lsdb.core.search import BoxSearch, ConeSearch, IndexSearch, OrderSearch, PolygonSearch


def assert_index_correct(cat):
    assert cat._ddf.index.name == SPATIAL_INDEX_COLUMN
    cat_comp = cat.compute()
    assert cat_comp.index.name == SPATIAL_INDEX_COLUMN
    npt.assert_array_equal(
        cat_comp.index.to_numpy(),
        compute_spatial_index(cat_comp["ra"].to_numpy(), cat_comp["dec"].to_numpy()),
    )


def test_read_hats(small_sky_order1_dir, small_sky_order1_hats_catalog, assert_divisions_are_correct):
    catalog = lsdb.read_hats(small_sky_order1_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hats_catalog.catalog_base_dir
    assert catalog.hc_structure.catalog_info.total_rows == len(catalog)
    assert catalog.get_healpix_pixels() == small_sky_order1_hats_catalog.get_healpix_pixels()
    assert len(catalog.compute().columns) == 8
    assert isinstance(catalog.compute(), npd.NestedFrame)
    assert_divisions_are_correct(catalog)
    assert_index_correct(catalog)


def test_read_hats_no_pandas(small_sky_order1_no_pandas_dir, assert_divisions_are_correct):
    catalog = lsdb.read_hats(small_sky_order1_no_pandas_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert len(catalog.compute().columns) == 8
    assert isinstance(catalog.compute(), npd.NestedFrame)
    assert_divisions_are_correct(catalog)
    assert_index_correct(catalog)


def test_read_hats_with_margin_extra_kwargs(small_sky_xmatch_dir, small_sky_xmatch_margin_dir):
    catalog = lsdb.read_hats(
        small_sky_xmatch_dir,
        margin_cache=small_sky_xmatch_margin_dir,
        columns=["ra", "dec"],
        filters=[("ra", ">", 300)],
        engine="pyarrow",
    )
    assert isinstance(catalog, lsdb.Catalog)
    filtered_cat = catalog.compute()
    assert all(catalog.columns == ["ra", "dec"])
    assert np.all(filtered_cat["ra"] > 300)

    margin = catalog.margin
    assert isinstance(margin, lsdb.MarginCatalog)
    filtered_margin = margin.compute()
    assert all(margin.columns == ["ra", "dec"])
    assert np.all(filtered_margin["ra"] > 300)


def test_read_hats_with_columns(small_sky_order1_dir):
    filter_columns = ["ra", "dec"]
    catalog = lsdb.read_hats(small_sky_order1_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    npt.assert_array_equal(catalog.compute().columns.to_numpy(), filter_columns)
    assert_index_correct(catalog)


def test_read_hats_no_pandas_with_columns(small_sky_order1_no_pandas_dir):
    filter_columns = ["ra", "dec"]
    catalog = lsdb.read_hats(small_sky_order1_no_pandas_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    npt.assert_array_equal(catalog.compute().columns.to_numpy(), filter_columns)
    assert_index_correct(catalog)


def test_read_hats_no_pandas_with_index_column(small_sky_order1_no_pandas_dir):
    filter_columns = ["ra", "dec", "_healpix_29"]
    catalog = lsdb.read_hats(small_sky_order1_no_pandas_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    assert_index_correct(catalog)
    npt.assert_array_equal(catalog.compute().columns.to_numpy(), filter_columns)


def test_read_hats_with_extra_kwargs(small_sky_order1_dir):
    catalog = lsdb.read_hats(small_sky_order1_dir, filters=[("ra", ">", 300)], engine="pyarrow")
    assert isinstance(catalog, lsdb.Catalog)
    assert np.greater(catalog.compute()["ra"].to_numpy(), 300).all()


def test_pixels_in_map_equal_catalog_pixels(small_sky_order1_dir, small_sky_order1_hats_catalog):
    catalog = lsdb.read_hats(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hats_catalog.get_healpix_pixels():
        catalog.get_partition(healpix_pixel.order, healpix_pixel.pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir):
    catalog = lsdb.read_hats(small_sky_order1_dir)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(small_sky_order1_dir, small_sky_order1_hats_catalog):
    catalog = lsdb.read_hats(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hats_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hats_catalog.catalog_base_dir, healpix_pixel
        )
        loaded_df = pd.read_parquet(parquet_path, dtype_backend="pyarrow")
        pd.testing.assert_frame_equal(partition_df, loaded_df)


def test_read_hats_specify_catalog_type(small_sky_catalog, small_sky_dir):
    catalog = lsdb.read_hats(small_sky_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog.compute())
    assert catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info
    assert isinstance(catalog.compute(), npd.NestedFrame)


def test_catalog_with_margin(
    small_sky_xmatch_dir, small_sky_xmatch_margin_dir, small_sky_xmatch_margin_catalog
):
    assert isinstance(small_sky_xmatch_margin_dir, Path)
    catalog = lsdb.read_hats(small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert isinstance(catalog.margin._ddf, nd.NestedFrame)
    assert (
        catalog.margin.hc_structure.catalog_info == small_sky_xmatch_margin_catalog.hc_structure.catalog_info
    )
    assert catalog.margin.get_healpix_pixels() == small_sky_xmatch_margin_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(catalog.margin.compute(), small_sky_xmatch_margin_catalog.compute())


def test_catalog_without_margin_is_none(small_sky_xmatch_dir):
    catalog = lsdb.read_hats(small_sky_xmatch_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.margin is None


def test_catalog_with_wrong_margin(small_sky_order1_dir, small_sky_order1_source_margin_dir):
    with pytest.raises(ValueError, match="must have the same schema"):
        lsdb.read_hats(small_sky_order1_dir, margin_cache=small_sky_order1_source_margin_dir)


def test_read_hats_subset_with_cone_search(small_sky_order1_dir, small_sky_order1_catalog):
    cone_search = ConeSearch(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering using catalog's cone_search
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering when calling `read_hats`
    cone_search_catalog_2 = lsdb.read_hats(small_sky_order1_dir, search_filter=cone_search)
    assert isinstance(cone_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert cone_search_catalog.get_healpix_pixels() == cone_search_catalog_2.get_healpix_pixels()
    pd.testing.assert_frame_equal(cone_search_catalog.compute(), cone_search_catalog_2.compute())


def test_read_hats_subset_with_box_search(small_sky_order1_dir, small_sky_order1_catalog):
    box_search = BoxSearch(ra=(300, 320), dec=(-40, -10))
    # Filtering using catalog's box_search
    box_search_catalog = small_sky_order1_catalog.box_search(ra=(300, 320), dec=(-40, -10))
    # Filtering when calling `read_hats`
    box_search_catalog_2 = lsdb.read_hats(small_sky_order1_dir, search_filter=box_search)
    assert isinstance(box_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert box_search_catalog.get_healpix_pixels() == box_search_catalog_2.get_healpix_pixels()
    pd.testing.assert_frame_equal(box_search_catalog.compute(), box_search_catalog_2.compute())


def test_read_hats_subset_with_polygon_search(small_sky_order1_dir, small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon_search = PolygonSearch(vertices)
    # Filtering using catalog's polygon_search
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    # Filtering when calling `read_hats`
    polygon_search_catalog_2 = lsdb.read_hats(small_sky_order1_dir, search_filter=polygon_search)
    assert isinstance(polygon_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert polygon_search_catalog.get_healpix_pixels() == polygon_search_catalog_2.get_healpix_pixels()
    pd.testing.assert_frame_equal(polygon_search_catalog.compute(), polygon_search_catalog_2.compute())


def test_read_hats_subset_with_index_search(
    small_sky_order1_dir,
    small_sky_order1_catalog,
    small_sky_order1_id_index_dir,
):
    catalog_index = hc.read_hats(small_sky_order1_id_index_dir)
    # Filtering using catalog's index_search
    index_search_catalog = small_sky_order1_catalog.index_search([700], catalog_index)
    # Filtering when calling `read_hats`
    index_search = IndexSearch([700], catalog_index)
    index_search_catalog_2 = lsdb.read_hats(small_sky_order1_dir, search_filter=index_search)
    assert isinstance(index_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert index_search_catalog.get_healpix_pixels() == index_search_catalog_2.get_healpix_pixels()


def test_read_hats_subset_with_order_search(small_sky_source_catalog, small_sky_source_dir):
    order_search = OrderSearch(min_order=1, max_order=2)
    # Filtering using catalog's order_search
    order_search_catalog = small_sky_source_catalog.order_search(min_order=1, max_order=2)
    # Filtering when calling `read_hats`
    order_search_catalog_2 = lsdb.read_hats(small_sky_source_dir, search_filter=order_search)
    assert isinstance(order_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert order_search_catalog.get_healpix_pixels() == order_search_catalog_2.get_healpix_pixels()


def test_read_hats_subset_no_partitions(small_sky_order1_dir, small_sky_order1_id_index_dir):
    with pytest.raises(ValueError, match="no coverage"):
        catalog_index = hc.read_hats(small_sky_order1_id_index_dir)
        index_search = IndexSearch([900], catalog_index)
        lsdb.read_hats(small_sky_order1_dir, search_filter=index_search)


def test_read_hats_with_margin_subset(
    small_sky_order1_source_dir, small_sky_order1_source_with_margin, small_sky_order1_source_margin_dir
):
    cone_search = ConeSearch(ra=315, dec=-66, radius_arcsec=20)
    # Filtering using catalog's cone_search
    cone_search_catalog = small_sky_order1_source_with_margin.cone_search(ra=315, dec=-66, radius_arcsec=20)
    # Filtering when calling `read_hats`
    cone_search_catalog_2 = lsdb.read_hats(
        small_sky_order1_source_dir,
        search_filter=cone_search,
        margin_cache=small_sky_order1_source_margin_dir,
    )
    assert isinstance(cone_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert cone_search_catalog.get_healpix_pixels() == cone_search_catalog_2.get_healpix_pixels()
    assert (
        cone_search_catalog.margin.get_healpix_pixels() == cone_search_catalog_2.margin.get_healpix_pixels()
    )


def test_read_hats_with_backend(small_sky_dir):
    # By default, the schema is backed by pyarrow
    default_catalog = lsdb.read_hats(small_sky_dir)
    assert all(isinstance(col_type, pd.ArrowDtype) for col_type in default_catalog.dtypes)
    # We can also pass it explicitly as an argument
    catalog = lsdb.read_hats(small_sky_dir, dtype_backend="pyarrow")
    assert catalog.dtypes.equals(default_catalog.dtypes)
    # Load data using a numpy-nullable types.
    catalog = lsdb.read_hats(small_sky_dir, dtype_backend="numpy_nullable")
    assert all(isinstance(col_type, ExtensionDtype) for col_type in catalog.dtypes)
    # The other option is to keep the original types. In this case they are numpy-backed.
    catalog = lsdb.read_hats(small_sky_dir, dtype_backend=None)
    assert all(isinstance(col_type, np.dtype) for col_type in catalog.dtypes)

    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert isinstance(catalog.compute(), npd.NestedFrame)


def test_read_hats_with_invalid_backend(small_sky_dir):
    with pytest.raises(ValueError, match="data type backend must be either"):
        lsdb.read_hats(small_sky_dir, dtype_backend="abc")


def test_read_hats_margin_catalog_subset(
    small_sky_order1_source_margin_dir, small_sky_order1_source_margin_catalog, assert_divisions_are_correct
):
    search_filter = ConeSearch(ra=315, dec=-60, radius_arcsec=10)
    margin = lsdb.read_hats(small_sky_order1_source_margin_dir, search_filter=search_filter)

    margin_info = margin.hc_structure.catalog_info
    small_sky_order1_source_margin_info = small_sky_order1_source_margin_catalog.hc_structure.catalog_info

    assert isinstance(margin, lsdb.MarginCatalog)
    assert margin_info.catalog_name == small_sky_order1_source_margin_info.catalog_name
    assert margin_info.primary_catalog == small_sky_order1_source_margin_info.primary_catalog
    assert margin_info.margin_threshold == small_sky_order1_source_margin_info.margin_threshold
    assert margin.get_healpix_pixels() == [
        HealpixPixel(1, 44),
        HealpixPixel(1, 45),
        HealpixPixel(1, 46),
        HealpixPixel(1, 47),
    ]
    assert_divisions_are_correct(margin)


def test_read_hats_margin_catalog_subset_is_empty(small_sky_order1_source_margin_dir):
    search_filter = ConeSearch(ra=100, dec=80, radius_arcsec=1)
    with pytest.raises(ValueError, match="empty catalog"):
        lsdb.read_hats(small_sky_order1_source_margin_dir, search_filter=search_filter)


def test_read_hats_schema_not_found(small_sky_no_metadata_dir):
    with pytest.raises(ValueError, match="catalog schema could not be loaded"):
        lsdb.read_hats(small_sky_no_metadata_dir)
