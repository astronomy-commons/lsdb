from pathlib import Path
from unittest.mock import call

import hats as hc
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
from hats.io.file_io import get_upath_for_protocol
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from upath import UPath

import lsdb
import lsdb.nested as nd
from lsdb.core.search.index_search import IndexSearch
from lsdb.core.search.region_search import BoxSearch, ConeSearch, OrderSearch, PolygonSearch


def test_read_hats(small_sky_order1_dir, small_sky_order1_hats_catalog, helpers):
    catalog = lsdb.open_catalog(small_sky_order1_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_base_dir == small_sky_order1_hats_catalog.catalog_base_dir
    assert catalog.hc_structure.catalog_info.total_rows == len(catalog)
    assert catalog.get_healpix_pixels() == small_sky_order1_hats_catalog.get_healpix_pixels()
    assert len(catalog.compute().columns) == 5
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)


def test_read_hats_collection_with_default_margin(
    small_sky_order1_collection_dir, small_sky_order1_catalog, small_sky_order1_margin_1deg_catalog, helpers
):
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir)

    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.name == catalog.hc_collection.main_catalog.catalog_name
    main_catalog_dir = small_sky_order1_collection_dir / small_sky_order1_catalog.name
    assert catalog.hc_structure.catalog_base_dir == main_catalog_dir
    assert catalog.hc_structure.catalog_info.total_rows == len(small_sky_order1_catalog)
    assert catalog.get_healpix_pixels() == small_sky_order1_catalog.get_healpix_pixels()
    assert len(catalog.compute().columns) == 5
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert catalog.margin.name == catalog.hc_collection.default_margin
    margin_catalog_dir = small_sky_order1_collection_dir / small_sky_order1_margin_1deg_catalog.name
    assert catalog.margin.hc_structure.catalog_base_dir == margin_catalog_dir
    assert catalog.margin.hc_structure.catalog_info.total_rows == len(small_sky_order1_margin_1deg_catalog)
    assert catalog.margin.get_healpix_pixels() == small_sky_order1_margin_1deg_catalog.get_healpix_pixels()
    assert len(catalog.margin.compute().columns) == 5
    assert isinstance(catalog.margin.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog.margin)
    helpers.assert_index_correct(catalog.margin)
    helpers.assert_schema_correct(catalog.margin)


def test_read_hats_collection_with_margin_name(
    small_sky_order1_collection_dir, small_sky_order1_margin_2deg_catalog, helpers
):
    margin_name = small_sky_order1_margin_2deg_catalog.name
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir, margin_cache=margin_name)

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert catalog.margin.name != catalog.hc_collection.default_margin
    margin_catalog_dir = small_sky_order1_collection_dir / small_sky_order1_margin_2deg_catalog.name
    assert catalog.margin.hc_structure.catalog_base_dir == margin_catalog_dir
    assert catalog.margin.hc_structure.catalog_info.total_rows == len(small_sky_order1_margin_2deg_catalog)
    assert catalog.margin.get_healpix_pixels() == small_sky_order1_margin_2deg_catalog.get_healpix_pixels()
    assert len(catalog.margin.compute().columns) == 5
    assert isinstance(catalog.margin.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog.margin)
    helpers.assert_index_correct(catalog.margin)
    helpers.assert_schema_correct(catalog.margin)


@pytest.mark.parametrize("path_type", [str, Path, UPath])
def test_read_hats_collection_with_margin_absolute_path(
    small_sky_order1_collection_dir,
    small_sky_order1_margin_2deg_dir,
    small_sky_order1_margin_2deg_catalog,
    path_type,
    helpers,
):
    margin_absolute_path = path_type(small_sky_order1_margin_2deg_dir)
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir, margin_cache=margin_absolute_path)

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert catalog.margin.name != catalog.hc_collection.default_margin
    assert str(catalog.margin.hc_structure.catalog_base_dir) == str(small_sky_order1_margin_2deg_dir)
    assert catalog.margin.hc_structure.catalog_info.total_rows == len(small_sky_order1_margin_2deg_catalog)
    assert catalog.margin.get_healpix_pixels() == small_sky_order1_margin_2deg_catalog.get_healpix_pixels()
    assert len(catalog.margin.compute().columns) == 5
    assert isinstance(catalog.margin.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog.margin)
    helpers.assert_index_correct(catalog.margin)
    helpers.assert_schema_correct(catalog.margin)


def test_read_hats_collection_with_extra_kwargs(small_sky_order1_collection_dir):
    catalog = lsdb.open_catalog(
        small_sky_order1_collection_dir, columns=["ra", "dec"], filters=[("ra", ">", 300)]
    )

    assert isinstance(catalog, lsdb.Catalog)
    assert all(catalog.columns == ["ra", "dec"])
    assert catalog.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    assert np.all(catalog.compute()["ra"] > 300)

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert all(catalog.margin.columns == ["ra", "dec"])
    assert catalog.margin.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    assert np.all(catalog.margin.compute()["ra"] > 300)


def test_read_hats_collection_nested_filters(small_sky_order1_collection_dir):
    """Tests that we appropriately handled nested pyarrow compute expressions,
    where all clauses are AND'd together"""
    catalog = lsdb.open_catalog(
        small_sky_order1_collection_dir,
        columns=["ra", "dec"],
        filters=[[("ra", ">", 300), ("dec", "<", -50)]],
    )

    assert isinstance(catalog, lsdb.Catalog)
    assert all(catalog.columns == ["ra", "dec"])
    assert catalog.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    catalog_contents = catalog.compute()
    assert np.all(catalog_contents["ra"] > 300)
    assert np.all(catalog_contents["dec"] < -50)

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert all(catalog.margin.columns == ["ra", "dec"])
    assert catalog.margin.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    catalog_contents = catalog.margin.compute()
    assert np.all(catalog_contents["ra"] > 300)
    assert np.all(catalog_contents["dec"] < -50)


def test_read_hats_collection_nested_filters_or(small_sky_order1_collection_dir):
    """Tests that we appropriately handled nested pyarrow compute expressions,
    where clauses are OR'd together"""
    catalog = lsdb.open_catalog(
        small_sky_order1_collection_dir,
        columns=["ra", "dec"],
        filters=[[("ra", ">", 340)], [("ra", "<", 290)]],
    )

    assert isinstance(catalog, lsdb.Catalog)
    assert all(catalog.columns == ["ra", "dec"])
    assert catalog.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    catalog_contents = catalog.compute()
    assert np.all(np.logical_or(catalog_contents["ra"] > 340, catalog_contents["ra"] < 290))

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert all(catalog.margin.columns == ["ra", "dec"])
    assert catalog.margin.hc_structure.schema.names == ["ra", "dec", SPATIAL_INDEX_COLUMN]
    catalog_contents = catalog.margin.compute()
    assert np.all(np.logical_or(catalog_contents["ra"] > 340, catalog_contents["ra"] < 290))


def test_read_hats_collection_empty_filters(small_sky_order1_collection_dir):
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir, filters=None)

    assert isinstance(catalog, lsdb.Catalog)
    assert all(catalog.columns == ["id", "ra", "dec", "ra_error", "dec_error"])

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert all(catalog.margin.columns == ["id", "ra", "dec", "ra_error", "dec_error"])

    catalog = lsdb.open_catalog(
        small_sky_order1_collection_dir / "small_sky_order1",
        margin_cache=small_sky_order1_collection_dir / "small_sky_order1_margin_1deg",
        filters=None,
    )

    assert isinstance(catalog, lsdb.Catalog)
    assert all(catalog.columns == ["id", "ra", "dec", "ra_error", "dec_error"])

    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert all(catalog.margin.columns == ["id", "ra", "dec", "ra_error", "dec_error"])


def test_read_hats_initializes_upath_once(
    small_sky_order1_source_dir, small_sky_order1_source_margin_dir, mocker
):
    mock_method = "hats.io.file_io.file_pointer.get_upath_for_protocol"
    mocked_upath_call = mocker.patch(mock_method, side_effect=get_upath_for_protocol)

    lsdb.open_catalog(small_sky_order1_source_dir, margin_cache=small_sky_order1_source_margin_dir).compute()

    expected_calls = [
        call.get_upath_for_protocol(small_sky_order1_source_dir),
        call.get_upath_for_protocol(small_sky_order1_source_margin_dir),
    ]
    mocked_upath_call.assert_has_calls(expected_calls)


def test_read_hats_default_cols(small_sky_order1_default_cols_dir, helpers):
    catalog = lsdb.open_catalog(small_sky_order1_default_cols_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_info.default_columns is not None
    assert np.all(catalog.columns == catalog.hc_structure.catalog_info.default_columns)
    assert np.all(catalog.compute().columns == catalog.hc_structure.catalog_info.default_columns)
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)
    helpers.assert_default_columns_in_columns(catalog)


def test_read_hats_default_cols_specify_cols(small_sky_order1_default_cols_dir, helpers):
    filter_columns = ["ra", "dec"]
    catalog = lsdb.open_catalog(small_sky_order1_default_cols_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_info.default_columns is not None
    assert np.all(catalog.columns == filter_columns)
    assert np.all(catalog.compute().columns == filter_columns)
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)
    helpers.assert_default_columns_in_columns(catalog)


def test_read_hats_default_cols_all_cols(small_sky_order1_default_cols_dir, helpers):
    expected_all_cols = ["id", "ra", "dec", "ra_error", "dec_error"]
    catalog = lsdb.open_catalog(small_sky_order1_default_cols_dir, columns="all")
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_info.default_columns is not None
    assert np.all(catalog.columns == expected_all_cols)
    assert np.all(catalog.compute().columns == expected_all_cols)
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)


def test_read_hats_default_cols_invalid_selector(small_sky_order1_default_cols_dir):
    with pytest.raises(TypeError):
        lsdb.open_catalog(small_sky_order1_default_cols_dir, columns="other")


def test_read_hats_no_pandas(small_sky_order1_no_pandas_dir, helpers):
    catalog = lsdb.open_catalog(small_sky_order1_no_pandas_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert len(catalog.compute().columns) == 5
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)


def test_read_hats_with_margin_extra_kwargs(small_sky_xmatch_dir, small_sky_xmatch_margin_dir):
    filter_columns = ("ra", "dec")
    catalog = lsdb.open_catalog(
        small_sky_xmatch_dir,
        margin_cache=small_sky_xmatch_margin_dir,
        columns=filter_columns,
        filters=[("ra", ">", 300)],
    )
    assert isinstance(catalog, lsdb.Catalog)
    filtered_cat = catalog.compute()
    assert list(catalog.columns) == list(filter_columns)
    assert np.all(filtered_cat["ra"] > 300)
    margin = catalog.margin
    assert isinstance(margin, lsdb.MarginCatalog)
    filtered_margin = margin.compute()
    assert list(margin.columns) == list(filter_columns)
    assert np.all(filtered_margin["ra"] > 300)


def test_read_hats_npix_alt_suffix(
    small_sky_npix_alt_suffix_dir, small_sky_npix_alt_suffix_hats_catalog, helpers
):
    catalog = lsdb.open_catalog(small_sky_npix_alt_suffix_dir)
    # Show that npix_suffix is not the standard ".parquet" but is still valid.
    catalog_npix_suffix = catalog.hc_structure.catalog_info.npix_suffix
    assert isinstance(catalog_npix_suffix, str)
    assert catalog_npix_suffix != ".parquet"
    assert catalog_npix_suffix == small_sky_npix_alt_suffix_hats_catalog.catalog_info.npix_suffix
    # Show that the catalog can be read as expected.
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_info.total_rows == len(catalog)
    assert len(catalog.compute().columns) == 5
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)


def test_read_hats_npix_as_dir(small_sky_npix_as_dir_dir, small_sky_npix_as_dir_hats_catalog, helpers):
    catalog = lsdb.open_catalog(small_sky_npix_as_dir_dir)
    # Show that npix_suffix indicates that Npix are directories and also matches the hats property.
    catalog_npix_suffix = catalog.hc_structure.catalog_info.npix_suffix
    assert catalog_npix_suffix == "/"
    assert catalog_npix_suffix == small_sky_npix_as_dir_hats_catalog.catalog_info.npix_suffix
    # Show that the catalog can be read as expected.
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert catalog.hc_structure.catalog_info.total_rows == len(catalog)
    assert len(catalog.compute().columns) == 5
    assert isinstance(catalog.compute(), npd.NestedFrame)
    helpers.assert_divisions_are_correct(catalog)
    helpers.assert_index_correct(catalog)


def test_read_hats_with_columns(small_sky_order1_dir, helpers):
    filter_columns = np.array(["ra", "dec"])
    catalog = lsdb.open_catalog(small_sky_order1_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    assert list(catalog.compute().columns) == list(filter_columns)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)


def test_read_hats_no_pandas_with_columns(small_sky_order1_no_pandas_dir, helpers):
    filter_columns = pd.Series(["ra", "dec"])
    catalog = lsdb.open_catalog(small_sky_order1_no_pandas_dir, columns=filter_columns)
    assert isinstance(catalog, lsdb.Catalog)
    assert list(catalog.compute().columns) == list(filter_columns)
    helpers.assert_index_correct(catalog)
    helpers.assert_schema_correct(catalog)


def test_read_hats_no_pandas_with_index_column(small_sky_order1_no_pandas_dir, helpers):
    # _healpix_29 might be included in the columns, and it has no side effect.
    catalog = lsdb.open_catalog(small_sky_order1_no_pandas_dir, columns=["_healpix_29"])
    assert "ra" in catalog.columns
    assert "dec" in catalog.columns
    assert isinstance(catalog, lsdb.Catalog)
    helpers.assert_index_correct(catalog)
    assert list(catalog.compute().columns) == list(["ra", "dec"])
    helpers.assert_schema_correct(catalog)


def test_read_hats_with_extra_kwargs(small_sky_order1_dir):
    catalog = lsdb.open_catalog(small_sky_order1_dir, filters=[("ra", ">", 300)])
    assert isinstance(catalog, lsdb.Catalog)
    assert np.greater(catalog.compute()["ra"].to_numpy(), 300).all()

    assert catalog.loading_config.make_query_url_params() == {"filters": "ra>300"}


def test_read_hats_with_mistaken_kwargs(small_sky_order1_dir, small_sky_xmatch_margin_dir):
    with pytest.raises(ValueError, match="Invalid keyword argument"):
        lsdb.open_catalog(small_sky_order1_dir, margins=small_sky_xmatch_margin_dir)


def test_pixels_in_map_equal_catalog_pixels(small_sky_order1_dir, small_sky_order1_hats_catalog):
    catalog = lsdb.open_catalog(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hats_catalog.get_healpix_pixels():
        catalog.get_partition(healpix_pixel.order, healpix_pixel.pixel)


def test_wrong_pixel_raises_value_error(small_sky_order1_dir):
    catalog = lsdb.open_catalog(small_sky_order1_dir)
    with pytest.raises(ValueError):
        catalog.get_partition(-1, -1)


def test_parquet_data_in_partitions_match_files(small_sky_order1_dir, small_sky_order1_hats_catalog):
    catalog = lsdb.open_catalog(small_sky_order1_dir)
    for healpix_pixel in small_sky_order1_hats_catalog.get_healpix_pixels():
        hp_order = healpix_pixel.order
        hp_pixel = healpix_pixel.pixel
        partition = catalog.get_partition(hp_order, hp_pixel)
        partition_df = partition.compute()
        parquet_path = hc.io.paths.pixel_catalog_file(
            small_sky_order1_hats_catalog.catalog_base_dir, healpix_pixel
        )
        loaded_df = pd.read_parquet(parquet_path, dtype_backend="pyarrow").set_index("_healpix_29")
        pd.testing.assert_frame_equal(partition_df, loaded_df)


def test_read_hats_specify_catalog_type(small_sky_catalog, small_sky_dir):
    catalog = lsdb.open_catalog(small_sky_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_catalog.compute())
    assert catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    assert catalog.hc_structure.catalog_info == small_sky_catalog.hc_structure.catalog_info
    assert isinstance(catalog.compute(), npd.NestedFrame)


def test_catalog_with_margin(
    small_sky_xmatch_dir, small_sky_xmatch_margin_dir, small_sky_xmatch_margin_catalog, helpers
):
    assert isinstance(small_sky_xmatch_margin_dir, Path)
    catalog = lsdb.open_catalog(small_sky_xmatch_dir, margin_cache=small_sky_xmatch_margin_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert isinstance(catalog.margin, lsdb.MarginCatalog)
    assert isinstance(catalog._ddf, nd.NestedFrame)
    assert isinstance(catalog.margin._ddf, nd.NestedFrame)
    assert (
        catalog.margin.hc_structure.catalog_info == small_sky_xmatch_margin_catalog.hc_structure.catalog_info
    )
    assert catalog.margin.get_healpix_pixels() == small_sky_xmatch_margin_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(catalog.margin.compute(), small_sky_xmatch_margin_catalog.compute())
    helpers.assert_schema_correct(catalog)
    helpers.assert_schema_correct(catalog.margin)


def test_catalog_without_margin_is_none(small_sky_xmatch_dir):
    catalog = lsdb.open_catalog(small_sky_xmatch_dir)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.margin is None


def test_catalog_with_wrong_margin(small_sky_order1_dir, small_sky_order1_source_margin_dir):
    with pytest.raises(ValueError, match="must have the same schema"):
        lsdb.open_catalog(small_sky_order1_dir, margin_cache=small_sky_order1_source_margin_dir)


def test_read_hats_pixel_search_radec(small_sky_order1_collection_dir):
    filtered_catalog = lsdb.open_catalog(
        small_sky_order1_collection_dir, search_filter=lsdb.PixelSearch.from_radec(ra=280.5, dec=-35.5)
    )
    assert filtered_catalog.get_healpix_pixels() == [HealpixPixel(1, 46)]
    assert filtered_catalog.margin is not None
    assert filtered_catalog.margin.get_healpix_pixels() == [HealpixPixel(1, 46)]


def test_read_hats_subset_with_cone_search(small_sky_order1_dir, small_sky_order1_catalog):
    cone_search = ConeSearch(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering using catalog's cone_search
    cone_search_catalog = small_sky_order1_catalog.cone_search(ra=0, dec=-80, radius_arcsec=20 * 3600)
    # Filtering when calling `read_hats`
    cone_search_catalog_2 = lsdb.open_catalog(small_sky_order1_dir, search_filter=cone_search)
    assert isinstance(cone_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert cone_search_catalog.get_healpix_pixels() == cone_search_catalog_2.get_healpix_pixels()
    pd.testing.assert_frame_equal(cone_search_catalog.compute(), cone_search_catalog_2.compute())


def test_read_hats_subset_with_box_search(small_sky_order1_dir, small_sky_order1_catalog):
    box_search = BoxSearch(ra=(300, 320), dec=(-40, -10))
    # Filtering using catalog's box_search
    box_search_catalog = small_sky_order1_catalog.box_search(ra=(300, 320), dec=(-40, -10))
    # Filtering when calling `read_hats`
    box_search_catalog_2 = lsdb.open_catalog(small_sky_order1_dir, search_filter=box_search)
    assert isinstance(box_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert box_search_catalog.get_healpix_pixels() == box_search_catalog_2.get_healpix_pixels()
    pd.testing.assert_frame_equal(box_search_catalog.compute(), box_search_catalog_2.compute())


@pytest.mark.sphgeom
def test_read_hats_subset_with_polygon_search(small_sky_order1_dir, small_sky_order1_catalog):
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon_search = PolygonSearch(vertices)
    # Filtering using catalog's polygon_search
    polygon_search_catalog = small_sky_order1_catalog.polygon_search(vertices)
    # Filtering when calling `read_hats`
    polygon_search_catalog_2 = lsdb.open_catalog(small_sky_order1_dir, search_filter=polygon_search)
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
    index_search_catalog = small_sky_order1_catalog.id_search(
        values={"id": 700}, index_catalogs={"id": catalog_index}
    )
    # Filtering when calling `read_hats`
    index_search = IndexSearch(values={"id": 700}, index_catalogs={"id": catalog_index})
    index_search_catalog_2 = lsdb.open_catalog(small_sky_order1_dir, search_filter=index_search)
    assert isinstance(index_search_catalog_2, lsdb.Catalog)
    # The partitions of the catalogs are equivalent
    assert index_search_catalog.get_healpix_pixels() == index_search_catalog_2.get_healpix_pixels()


def test_read_hats_subset_with_order_search(small_sky_source_catalog, small_sky_source_dir):
    order_search = OrderSearch(min_order=1, max_order=2)
    # Filtering using catalog's order_search
    order_search_catalog = small_sky_source_catalog.order_search(min_order=1, max_order=2)
    results1 = order_search_catalog.compute()
    # Filtering when calling `read_hats`
    order_search_catalog_2 = lsdb.open_catalog(
        small_sky_source_dir, search_filter=order_search, columns=["object_id", "object_ra", "object_dec"]
    )
    assert isinstance(order_search_catalog_2, lsdb.Catalog)
    results2 = order_search_catalog_2.compute()
    assert len(results1) == len(results2)
    # The partitions of the catalogs are equivalent
    assert order_search_catalog.get_healpix_pixels() == order_search_catalog_2.get_healpix_pixels()


def test_read_hats_subset_no_partitions(small_sky_order1_dir, small_sky_order1_id_index_dir):
    with pytest.raises(ValueError, match="no coverage"):
        catalog_index = hc.read_hats(small_sky_order1_id_index_dir)
        index_search = IndexSearch(values={"id": 900}, index_catalogs={"id": catalog_index})
        lsdb.open_catalog(small_sky_order1_dir, search_filter=index_search)


def test_read_hats_with_margin_subset(
    small_sky_order1_source_dir, small_sky_order1_source_with_margin, small_sky_order1_source_margin_dir
):
    cone_search = ConeSearch(ra=315, dec=-66, radius_arcsec=20)
    # Filtering using catalog's cone_search
    cone_search_catalog = small_sky_order1_source_with_margin.cone_search(ra=315, dec=-66, radius_arcsec=20)
    # Filtering when calling `read_hats`
    cone_search_catalog_2 = lsdb.open_catalog(
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


def test_read_hats_margin_catalog_subset(
    small_sky_order1_source_margin_dir, small_sky_order1_source_margin_catalog, helpers
):
    search_filter = ConeSearch(ra=315, dec=-60, radius_arcsec=10)
    margin = lsdb.open_catalog(small_sky_order1_source_margin_dir, search_filter=search_filter)

    margin_info = margin.hc_structure.catalog_info
    small_sky_order1_source_margin_info = small_sky_order1_source_margin_catalog.hc_structure.catalog_info

    assert isinstance(margin, lsdb.MarginCatalog)
    assert margin_info.catalog_name == small_sky_order1_source_margin_info.catalog_name
    assert margin_info.primary_catalog == small_sky_order1_source_margin_info.primary_catalog
    assert margin_info.margin_threshold == small_sky_order1_source_margin_info.margin_threshold
    assert margin.get_healpix_pixels() == [
        HealpixPixel(0, 8),
        HealpixPixel(1, 44),
        HealpixPixel(1, 45),
        HealpixPixel(1, 46),
        HealpixPixel(1, 47),
    ]
    helpers.assert_divisions_are_correct(margin)


def test_read_hats_margin_catalog_subset_is_empty(small_sky_order1_source_margin_dir):
    search_filter = ConeSearch(ra=100, dec=80, radius_arcsec=1)
    with pytest.raises(ValueError, match="no coverage"):
        lsdb.open_catalog(small_sky_order1_source_margin_dir, search_filter=search_filter)


def test_read_hats_map_catalog(test_data_dir):
    margin_catalog = lsdb.open_catalog(test_data_dir / "square_map")
    assert len(margin_catalog.get_healpix_pixels()) == 12
    assert len(margin_catalog._ddf_pixel_map) == 12
    assert len(margin_catalog.compute()) == 12
    assert len(margin_catalog.hc_structure.pixel_tree) == 12


def test_read_hats_schema_not_found(small_sky_no_metadata_dir):
    with pytest.raises(ValueError, match="catalog schema could not be loaded"):
        lsdb.open_catalog(small_sky_no_metadata_dir)


def test_all_columns_read_columns(small_sky_order1_dir, small_sky_order1_catalog):
    cat = lsdb.open_catalog(small_sky_order1_dir, columns=["ra", "dec"])
    assert len(cat.columns) < len(cat.all_columns)
    assert np.all(cat.all_columns == small_sky_order1_catalog.columns)


def test_original_schema_read_columns(small_sky_order1_dir, small_sky_order1_catalog):
    cat = lsdb.open_catalog(small_sky_order1_dir, columns=["ra", "dec"])
    assert len(cat.original_schema) == len(small_sky_order1_catalog.hc_structure.schema)
    for field in cat.original_schema:
        assert field in small_sky_order1_catalog.hc_structure.schema
    assert cat.original_schema == small_sky_order1_catalog.original_schema


def test_read_nested_column_selection(small_sky_with_nested_sources_dir):
    cat = lsdb.open_catalog(
        small_sky_with_nested_sources_dir, columns=["ra", "dec", "sources.source_id", "sources.source_ra"]
    )
    assert np.all(cat.columns == ["ra", "dec", "sources"])
    assert cat.dtypes["sources"].field_names == ["source_id", "source_ra"]
    computed = cat.compute()
    assert np.all(computed.columns == ["ra", "dec", "sources"])
    assert np.all(computed["sources"].iloc[0].columns == ["source_id", "source_ra"])


def test_read_nested_column_selection_errors(small_sky_with_nested_sources_dir):
    with pytest.raises(KeyError):
        lsdb.open_catalog(
            small_sky_with_nested_sources_dir, columns=["ra", "dec", "sources.source_id", "sources.wrong"]
        )
    with pytest.raises(KeyError):
        lsdb.open_catalog(
            small_sky_with_nested_sources_dir,
            columns=["ra", "wrong", "sources.source_id", "sources.source_ra"],
        )
    with pytest.raises(KeyError):
        lsdb.open_catalog(
            small_sky_with_nested_sources_dir, columns=["ra", "dec", "wrong.source_id", "sources.source_ra"]
        )
