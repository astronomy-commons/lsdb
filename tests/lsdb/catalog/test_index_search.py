from hipscat.catalog.index.index_catalog import IndexCatalog


def test_index_search(small_sky_order1_catalog, small_sky_order1_id_index_dir, assert_divisions_are_correct):
    catalog_index = IndexCatalog.read_from_hipscat(small_sky_order1_id_index_dir)

    index_search_catalog = small_sky_order1_catalog.index_search([900], catalog_index)
    index_search_df = index_search_catalog.compute()
    assert len(index_search_df) == 0
    assert_divisions_are_correct(index_search_catalog)

    index_search_catalog = small_sky_order1_catalog.index_search(["700"], catalog_index)
    index_search_df = index_search_catalog.compute()
    assert len(index_search_df) == 0
    assert_divisions_are_correct(index_search_catalog)

    index_search_catalog = small_sky_order1_catalog.index_search([700], catalog_index)
    index_search_df = index_search_catalog.compute()
    assert len(index_search_df) == 1
    assert_divisions_are_correct(index_search_catalog)
