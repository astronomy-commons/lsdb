import hipscat as hc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from hipscat.catalog.index.index_catalog import IndexCatalog

import lsdb
from lsdb.core.search import BoxSearch, ConeSearch, IndexSearch, OrderSearch, PolygonSearch


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


def test_read_hipscat_subset_with_cone_search(small_sky_order1_dir, small_sky_order1_catalog):
    cone_search = ConeSearch(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering using catalog's cone_search
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering when calling `read_hipscat`
    cone_search_catalog_2 = lsdb.read_hipscat(small_sky_order1_dir, search_filter=cone_search)
    assert isinstance(cone_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert cone_search_catalog.get_healpix_pixels() == cone_search_catalog_2.get_healpix_pixels()


def test_read_hipscat_subset_with_box_search(small_sky_order1_dir, small_sky_order1_catalog):
    box_search = BoxSearch(ra=(0, 10), dec=(-20, 10))
    # Filtering using catalog's box_search
    box_search_catalog = small_sky_order1_catalog.box(ra=(0, 10), dec=(-20, 10))
    # Filtering when calling `read_hipscat`
    box_search_catalog_2 = lsdb.read_hipscat(small_sky_order1_dir, search_filter=box_search)
    assert isinstance(box_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert box_search_catalog.get_healpix_pixels() == box_search_catalog_2.get_healpix_pixels()


def test_read_hipscat_subset_with_polygon_search(small_sky_order1_dir, small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon_search = PolygonSearch(vertices)
    # Filtering using catalog's polygon_search
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    # Filtering when calling `read_hipscat`
    polygon_search_catalog_2 = lsdb.read_hipscat(small_sky_order1_dir, search_filter=polygon_search)
    assert isinstance(polygon_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert polygon_search_catalog.get_healpix_pixels() == polygon_search_catalog_2.get_healpix_pixels()


def test_read_hipscat_subset_with_index_search(
    small_sky_order1_dir,
    small_sky_order1_catalog,
    small_sky_order1_id_index_dir,
):
    catalog_index = IndexCatalog.read_from_hipscat(small_sky_order1_id_index_dir)
    # Filtering using catalog's index_search
    index_search_catalog = small_sky_order1_catalog.index_search([700], catalog_index)
    # Filtering when calling `read_hipscat`
    index_search = IndexSearch([700], catalog_index)
    index_search_catalog_2 = lsdb.read_hipscat(small_sky_order1_dir, search_filter=index_search)
    assert isinstance(index_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert index_search_catalog.get_healpix_pixels() == index_search_catalog_2.get_healpix_pixels()


def test_read_hipscat_subset_with_order_search(small_sky_source_catalog, small_sky_source_dir):
    order_search = OrderSearch(min_order=1, max_order=2)
    # Filtering using catalog's order_search
    order_search_catalog = small_sky_source_catalog.order_search(min_order=1, max_order=2)
    # Filtering when calling `read_hipscat`
    order_search_catalog_2 = lsdb.read_hipscat(small_sky_source_dir, search_filter=order_search)
    assert isinstance(order_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert order_search_catalog.get_healpix_pixels() == order_search_catalog_2.get_healpix_pixels()


def test_read_hipscat_subset_no_partitions(small_sky_order1_dir, small_sky_order1_id_index_dir):
    with pytest.raises(ValueError, match="no partitions"):
        catalog_index = IndexCatalog.read_from_hipscat(small_sky_order1_id_index_dir)
        index_search = IndexSearch([900], catalog_index)
        lsdb.read_hipscat(small_sky_order1_dir, search_filter=index_search)
