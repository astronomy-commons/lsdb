import shutil

import hats
import nested_dask as nd
import nested_pandas as npd
import pandas as pd
import pytest
from hats.catalog.dataset.collection_properties import CollectionProperties

from lsdb import read_hats


def test_index_search(small_sky_order1_catalog, small_sky_order1_id_index_dir, helpers):
    catalog_index = hats.read_hats(small_sky_order1_id_index_dir)
    # Searching for an object that does not exist
    index_search_catalog = small_sky_order1_catalog.index_search([900], catalog_index)
    assert isinstance(index_search_catalog._ddf, nd.NestedFrame)
    index_search_df = index_search_catalog.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 0
    helpers.assert_divisions_are_correct(index_search_catalog)
    # Searching for an object that exists
    index_search_catalog = small_sky_order1_catalog.index_search([700], catalog_index)
    index_search_df = index_search_catalog.compute()
    assert len(index_search_df) == 1
    helpers.assert_divisions_are_correct(index_search_catalog)


def test_index_search_coarse_versus_fine(small_sky_order1_catalog, small_sky_order1_id_index_dir):
    catalog_index = hats.read_hats(small_sky_order1_id_index_dir)
    coarse_index_search = small_sky_order1_catalog.index_search([700], catalog_index, fine=False)
    fine_index_search = small_sky_order1_catalog.index_search([700], catalog_index)
    assert coarse_index_search.get_healpix_pixels() == fine_index_search.get_healpix_pixels()
    assert coarse_index_search._ddf.npartitions == fine_index_search._ddf.npartitions
    assert len(coarse_index_search.compute()) > len(fine_index_search.compute())


def test_id_search_with_default_field(small_sky_order1_collection_catalog, helpers):
    # Searching by the default index field "id"
    assert small_sky_order1_collection_catalog.hc_collection.default_index_field == "id"

    # Searching for an object that does not exist
    id_search_catalog = small_sky_order1_collection_catalog.id_search([900])
    assert isinstance(id_search_catalog._ddf, nd.NestedFrame)
    id_search_df = id_search_catalog.compute()
    assert isinstance(id_search_df, npd.NestedFrame)
    assert len(id_search_df) == 0
    helpers.assert_divisions_are_correct(id_search_catalog)

    # Searching for an object that exists
    id_search_catalog = small_sky_order1_collection_catalog.id_search([700])
    id_search_df = id_search_catalog.compute()
    assert len(id_search_df) == 1
    helpers.assert_divisions_are_correct(id_search_catalog)

    # Called with a single value instead of a list
    id_search_df_single_value = small_sky_order1_collection_catalog.id_search(700).compute()
    pd.testing.assert_frame_equal(id_search_df, id_search_df_single_value)


def test_id_search_with_field(small_sky_order1_collection_dir, tmp_path, helpers):
    # Searching by the index field "id" when it is not the default index

    # Copy the collection to a temporary directory so that we can modify its properties
    collection_base_dir = tmp_path / "collection"
    shutil.copytree(small_sky_order1_collection_dir, collection_base_dir)
    assert collection_base_dir.exists()
    collection_properties = CollectionProperties.read_from_dir(collection_base_dir)
    collection_properties.default_index = None
    collection_properties.to_properties_file(collection_base_dir)

    modified_catalog = read_hats(collection_base_dir)
    assert modified_catalog.hc_collection.all_indexes == {"id": "small_sky_order1_id_index"}
    assert modified_catalog.hc_collection.default_index_field is None

    # Searching for an object that does not exist
    id_search_catalog = modified_catalog.id_search([900], "id")
    assert isinstance(id_search_catalog._ddf, nd.NestedFrame)
    id_search_df = id_search_catalog.compute()
    assert isinstance(id_search_df, npd.NestedFrame)
    assert len(id_search_df) == 0
    helpers.assert_divisions_are_correct(id_search_catalog)

    # Searching for an object that exists
    id_search_catalog = modified_catalog.id_search([700], "id")
    id_search_df = id_search_catalog.compute()
    assert len(id_search_df) == 1
    helpers.assert_divisions_are_correct(id_search_catalog)

    # The default index is None; not specifying the "id" column raises an error
    with pytest.raises(ValueError, match="no default index field set"):
        modified_catalog.id_search([900])


def test_id_search_with_field_that_has_no_index_catalog(small_sky_order1_collection_catalog):
    # Searching by the index field "name" which is not specified
    assert "name" not in small_sky_order1_collection_catalog.hc_collection.all_indexes
    with pytest.raises(ValueError, match="not specified"):
        small_sky_order1_collection_catalog.id_search(["obj_1"], id_column="name")


def test_id_search_with_index_catalog_of_invalid_type(small_sky_order1_collection_dir, tmp_path):
    # The index catalog for "id" is of type `MarginCatalog` instead of `IndexCatalog`

    # Copy the collection to a temporary directory so that we can modify its properties
    collection_base_dir = tmp_path / "collection"
    shutil.copytree(small_sky_order1_collection_dir, collection_base_dir)
    assert collection_base_dir.exists()
    collection_properties = CollectionProperties.read_from_dir(collection_base_dir)
    collection_properties.all_indexes = {"id": "small_sky_order1_margin_1deg"}
    collection_properties.to_properties_file(collection_base_dir)

    modified_catalog = read_hats(collection_base_dir)
    with pytest.raises(TypeError, match="index is not of type"):
        modified_catalog.id_search([900], id_column="id")


def test_id_search_with_standalone_catalog(small_sky_order1_catalog):
    with pytest.raises(NotImplementedError):
        small_sky_order1_catalog.id_search([900])
