import shutil

import hats
import nested_pandas as npd
import pytest
from hats.catalog.dataset.collection_properties import CollectionProperties

import lsdb
from lsdb.core.search.index_search import IndexSearch


def test_id_search_with_single_field(
    small_sky_order1_source_collection_catalog,
):
    assert "object_id" in small_sky_order1_source_collection_catalog.hc_collection.all_indexes

    # Searching for the sources of an object that exists
    cat = small_sky_order1_source_collection_catalog.id_search(values={"object_id": 810})
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert len(index_search_df) == 131
    assert all(index_search_df["object_id"] == 810)

    # Searching for the sources of an object that does not exist
    cat = small_sky_order1_source_collection_catalog.id_search(values={"object_id": 900})
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 0


def test_id_search_with_multiple_fields(
    small_sky_order1_source_collection_catalog,
):
    cat = small_sky_order1_source_collection_catalog.id_search(values={"object_id": 810, "band": "r"})
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 17
    assert all(index_search_df["object_id"] == 810)
    assert all(index_search_df["band"] == "r")


def test_id_search_with_index_catalog_path(
    small_sky_order1_source_with_margin, small_sky_order1_source_object_id_index_dir
):
    cat = small_sky_order1_source_with_margin.id_search(
        values={"object_id": 810}, index_catalogs={"object_id": small_sky_order1_source_object_id_index_dir}
    )
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 131


def test_id_search_with_index_catalog(
    small_sky_order1_source_with_margin, small_sky_order1_source_object_id_index_dir
):
    id_index = hats.read_hats(small_sky_order1_source_object_id_index_dir)
    index_search_catalog = small_sky_order1_source_with_margin.id_search(
        values={"object_id": 810}, index_catalogs={"object_id": id_index}
    )
    assert isinstance(index_search_catalog.meta, npd.NestedFrame)
    index_search_df = index_search_catalog.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 131


def test_id_search_with_some_explicit_index_catalogs(
    small_sky_order1_source_collection_dir, small_sky_order1_source_band_index_dir, tmp_path
):
    # Copy the collection to a temporary directory
    collection_base_dir = tmp_path / "collection"
    shutil.copytree(small_sky_order1_source_collection_dir, collection_base_dir)
    assert collection_base_dir.exists()

    # Remove the band index from the collection properties
    collection_properties = CollectionProperties.read_from_dir(collection_base_dir)
    collection_properties.all_indexes = {"object_id": "small_sky_order1_source_object_id_index"}
    collection_properties.to_properties_file(collection_base_dir)

    collection = lsdb.open_catalog(collection_base_dir)
    cat = collection.id_search(
        values={"object_id": 810, "band": "r"},
        index_catalogs={"band": small_sky_order1_source_band_index_dir},
    )
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert len(index_search_df) == 17
    assert all(index_search_df["object_id"] == 810)
    assert all(index_search_df["band"] == "r")


def test_id_search_coarse(small_sky_order1_source_collection_catalog):
    fine_search = small_sky_order1_source_collection_catalog.id_search(values={"object_id": 810})
    coarse_search = small_sky_order1_source_collection_catalog.id_search(
        values={"object_id": 810}, fine=False
    )
    assert coarse_search.get_healpix_pixels() == fine_search.get_healpix_pixels()
    assert coarse_search._operation.healpix_pixels == fine_search._operation.healpix_pixels
    assert len(coarse_search.compute()) > len(fine_search.compute())


def test_id_search_when_no_index_for_field_is_available(small_sky_order1_source_collection_catalog):
    with pytest.raises(ValueError, match="`source_id` is not specified"):
        small_sky_order1_source_collection_catalog.id_search(values={"source_id": 70003})


def test_index_search_with_index_catalog_of_invalid_type(
    small_sky_order1_source_collection_catalog, small_sky_order1_source_margin_dir
):
    margin_catalog = hats.read_hats(small_sky_order1_source_margin_dir)
    with pytest.raises(TypeError, match="`band` is not of type `HCIndexCatalog`"):
        small_sky_order1_source_collection_catalog.id_search(
            values={"object_id": 810, "band": "r"},
            index_catalogs={"band": margin_catalog},
        )


def test_index_search_with_mismatching_fields(
    small_sky_order1_source_object_id_index_dir, small_sky_order1_source_band_index_dir
):
    with pytest.raises(ValueError, match="mismatch between the queried fields"):
        # "object_id" is missing in the `index_catalogs` mapping
        IndexSearch(
            values={"object_id": 100}, index_catalogs={"band": small_sky_order1_source_band_index_dir}
        )
    IndexSearch(
        values={"object_id": 100},
        index_catalogs={
            "object_id": small_sky_order1_source_object_id_index_dir,
            "band": small_sky_order1_source_band_index_dir,
        },
    )


def test_index_search_with_no_values(small_sky_order1_source_collection_catalog):
    # No values provided, should raise ValueError
    with pytest.raises(ValueError, match="No values specified for search."):
        small_sky_order1_source_collection_catalog.id_search(values={})


def test_id_search_with_list_of_values(small_sky_order1_source_collection_catalog):
    # Searching with object_id as a list
    cat = small_sky_order1_source_collection_catalog.id_search(values={"object_id": [810, 811]})
    assert isinstance(cat.meta, npd.NestedFrame)
    index_search_df = cat.compute()
    assert len(index_search_df) == 262  # 131 each
    assert all(index_search_df["object_id"].isin([810, 811]))

    # Searching with a list containing a non-existent object_id works as if the value was not in the list
    cat_plus_non_existent = small_sky_order1_source_collection_catalog.id_search(
        values={"object_id": [810, 811, 900]}
    )
    assert isinstance(cat_plus_non_existent.meta, npd.NestedFrame)
    index_search_df_plus_non_existent = cat_plus_non_existent.compute()
    assert len(index_search_df_plus_non_existent) == 262  # 131 each
    assert all(index_search_df_plus_non_existent["object_id"].isin([810, 811]))

    # ValueError when two or more columns contain list values
    with pytest.raises(ValueError, match="Only one column may contain a list of values."):
        small_sky_order1_source_collection_catalog.id_search(
            values={"object_id": [810, 811], "band": ["r", "g"]}
        )

    # Search works when only one column contains list values
    cat_with_only_one_list = small_sky_order1_source_collection_catalog.id_search(
        values={"object_id": [810, 811], "band": "r"}
    )
    assert isinstance(cat_with_only_one_list.meta, npd.NestedFrame)
    index_search_df_with_only_one_list = cat_with_only_one_list.compute()
    assert len(index_search_df_with_only_one_list) == 47  # 17 for object_id=810 + 30 for object_id=811
    assert all(index_search_df_with_only_one_list["object_id"].isin([810, 811]))
    assert all(index_search_df_with_only_one_list["band"] == "r")
