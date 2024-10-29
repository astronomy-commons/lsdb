import nested_dask as nd
import nested_pandas as npd
from hats import read_hats


def test_index_search(small_sky_order1_catalog, small_sky_order1_id_index_dir, assert_divisions_are_correct):
    catalog_index = read_hats(small_sky_order1_id_index_dir)
    # Searching for an object that does not exist
    index_search_catalog = small_sky_order1_catalog.index_search([900], catalog_index)
    assert isinstance(index_search_catalog._ddf, nd.NestedFrame)
    index_search_df = index_search_catalog.compute()
    assert isinstance(index_search_df, npd.NestedFrame)
    assert len(index_search_df) == 0
    assert_divisions_are_correct(index_search_catalog)
    # Searching for an object that exists
    index_search_catalog = small_sky_order1_catalog.index_search([700], catalog_index)
    index_search_df = index_search_catalog.compute()
    assert len(index_search_df) == 1
    assert_divisions_are_correct(index_search_catalog)


def test_index_search_coarse_versus_fine(small_sky_order1_catalog, small_sky_order1_id_index_dir):
    catalog_index = read_hats(small_sky_order1_id_index_dir)
    coarse_index_search = small_sky_order1_catalog.index_search([700], catalog_index, fine=False)
    fine_index_search = small_sky_order1_catalog.index_search([700], catalog_index)
    assert coarse_index_search.get_healpix_pixels() == fine_index_search.get_healpix_pixels()
    assert coarse_index_search._ddf.npartitions == fine_index_search._ddf.npartitions
    assert len(coarse_index_search.compute()) > len(fine_index_search.compute())
