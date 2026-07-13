import warnings

import pytest

import lsdb


def _assert_statistics(catalog, *, shown):
    """Assert whether the statistics table is rendered in both repr forms.

    When shown, the "min..max" table replaces the "..." placeholders; when hidden, the
    plain "..." placeholder table is rendered.
    """
    assert ("..." not in repr(catalog)) == shown
    assert ("..." not in catalog._repr_html_()) == shown


@pytest.mark.parametrize("repr_method", ["__repr__", "_repr_html_"])
def test_catalog_repr(small_sky_order1_catalog, repr_method):
    text = getattr(small_sky_order1_catalog, repr_method)()
    assert small_sky_order1_catalog.name in text
    assert str(small_sky_order1_catalog.get_ordered_healpix_pixels()[0]) in text
    assert str(small_sky_order1_catalog.get_ordered_healpix_pixels()[-1]) in text
    assert "available columns in the catalog have been loaded" in text
    assert "estimated size of" in text


@pytest.mark.parametrize("repr_method", ["__repr__", "_repr_html_"])
def test_catalog_text_repr_empty(small_sky_order1_catalog, repr_method):
    pixel_search = lsdb.PixelSearch.from_radec(80.0, 33.0)
    cat = small_sky_order1_catalog.search(pixel_search)
    text = getattr(cat, repr_method)()
    assert cat.name in text
    assert "Empty Catalog" in text
    assert "npartitions=0" in text
    assert "available columns in the catalog have been loaded" in text
    assert "estimated size of 0.0 Bytes" in text


def test_repr_mimebundle(small_sky_order1_dir):
    # Notebook display requests both text/plain and text/html.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)
    bundle = catalog._repr_mimebundle_()
    assert set(bundle) == {"text/plain", "text/html"}
    assert bundle["text/plain"] == repr(catalog)
    assert bundle["text/html"] == catalog._repr_html_()


def test_repr_statistics(small_sky_order1_dir):
    # By default, the repr does not show statistics.
    catalog = lsdb.open_catalog(small_sky_order1_dir)
    assert catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=False)
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)
    _assert_statistics(catalog, shown=True)


def test_shows_statistics_when_columns_are_selected(small_sky_order1_dir):
    # Selecting columns does not change the rows, so the on-disk statistics remain valid.
    catalog = lsdb.open_catalog(small_sky_order1_dir, columns=["ra", "dec"], show_statistics=True)
    assert list(catalog.columns) == ["ra", "dec"]
    assert catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=True)

    # Same when the column selection happens a posteriori.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)[["ra", "dec"]]
    assert list(catalog.columns) == ["ra", "dec"]
    assert catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=True)


def test_shows_statistics_with_coarse_search(small_sky_order1_dir):
    # A coarse spatial search retains whole partitions, so the on-disk statistics remain valid.
    catalog = lsdb.open_catalog(
        small_sky_order1_dir,
        search_filter=lsdb.ConeSearch(0, -80, 20 * 3600, fine=False),
        show_statistics=True,
    )
    assert catalog._operation.preserves_pixel_stats
    # The search adds MOC pruning filters, but those are not user-provided.
    assert not catalog.loading_config.user_provided_filters
    _assert_statistics(catalog, shown=True)

    # Same when the coarse search happens a posteriori.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)
    catalog = catalog.cone_search(0, -80, 20 * 3600, fine=False)
    assert catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=True)


def test_repr_does_not_warn_about_modified_catalog(small_sky_order1_dir):
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)
    catalog = catalog.cone_search(0, -80, 20 * 3600, fine=False)
    # A coarse search modifies the hats catalog, setting `total_rows` to None.
    assert catalog.hc_structure.catalog_info.total_rows is None
    # However, the pixel statistics are still preserved.
    assert catalog._operation.preserves_pixel_stats
    # The representation should show the statistics, and not raise a
    # "Results may be inaccurate" warning to the user.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _assert_statistics(catalog, shown=True)
    assert len(caught) == 0


def test_hides_statistics_with_fine_search(small_sky_order1_dir):
    # A fine search filters rows within partitions, so the statistics are not preserved.
    catalog = lsdb.open_catalog(
        small_sky_order1_dir,
        search_filter=lsdb.ConeSearch(0, -80, 20 * 3600),
        show_statistics=True,
    )
    assert not catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=False)

    # Same when the fine search happens a posteriori.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)
    catalog = catalog.cone_search(0, -80, 20 * 3600)
    assert not catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=False)


def test_hides_statistics_when_user_filters_applied(small_sky_order1_dir):
    # User-provided row `filters` drop rows, so the statistics are not preserved.
    catalog = lsdb.open_catalog(small_sky_order1_dir, filters=[("id", ">", 700)], show_statistics=True)
    assert catalog.loading_config.user_provided_filters
    assert catalog.hc_structure.catalog_info.total_rows is None
    _assert_statistics(catalog, shown=False)


def test_hides_statistics_when_modified_by_map_partitions(small_sky_order1_dir):
    # Operations involving map partitions do not preserve statistics.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True).query("id > 700")
    assert not catalog._operation.preserves_pixel_stats
    _assert_statistics(catalog, shown=False)


def test_hides_statistics_for_in_memory_catalog():
    # In-memory catalogs have no on-disk statistics.
    catalog = lsdb.generate_catalog(100, 1, seed=1)
    assert catalog.loading_config is None
    _assert_statistics(catalog, shown=False)


def test_repr_when_statistics_unavailable(small_sky_order1_dir, monkeypatch):
    # If the statistics cannot be loaded, the repr will fall back to the "..." placeholder table.
    catalog = lsdb.open_catalog(small_sky_order1_dir, show_statistics=True)

    def _not_found(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(catalog, "per_partition_statistics", _not_found)
    _assert_statistics(catalog, shown=False)


def test_repr_data_with_no_columns(small_sky_order1_dir):
    catalog = lsdb.open_catalog(small_sky_order1_dir)[[]]
    assert not list(catalog.columns)
    data = catalog._repr_data()
    assert not list(data.columns)
    assert len(data.index) == catalog.npartitions
    assert repr(catalog).startswith(f"lsdb Catalog {catalog.name}:")


def test_shows_statistics_except_for_nested_columns(small_sky_with_nested_sources_dir):
    # Nested columns have no per-column on-disk statistics, so they render the "..."
    # placeholder while every non-nested column shows its min..max range.
    catalog = lsdb.open_catalog(small_sky_with_nested_sources_dir, show_statistics=True)
    assert catalog._operation.preserves_pixel_stats
    assert catalog.nested_columns == ["sources"]

    # The dtype/type header is the first row; the remaining rows hold the per-pixel values.
    pixel_rows = catalog._repr_data().iloc[1:]
    assert (pixel_rows["sources"] == "...").all()
    for col in [c for c in pixel_rows.columns if c not in catalog.nested_columns]:
        assert not (pixel_rows[col] == "...").any()

    # A stats request for only nested columns has nothing to read.
    assert catalog._get_repr_pixel_stats(catalog.nested_columns) is None
