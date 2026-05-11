import sys

import pandas as pd
import pytest


def test_import_error_without_lancedb(monkeypatch):
    """Importing to_lance without lancedb installed raises a helpful ImportError."""
    # pylint: disable=import-outside-toplevel
    from lsdb.io.to_lance import to_lance  # noqa: F401

    monkeypatch.setitem(sys.modules, "lancedb", None)

    # Call to_lance with dummy arguments to trigger ImportError
    with pytest.raises(ImportError, match="to_lance requires the `lancedb` package"):
        to_lance(None, base_catalog_path="/tmp/does_not_matter")


def test_to_lance_writes_dataset(small_sky_catalog, tmp_path):
    """Basic write: dataset exists and row count matches."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_columns_match(small_sky_catalog, tmp_path):
    """All catalog columns (including the spatial index) appear in the Lance dataset."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    lance_columns = set(tbl.schema.names)

    computed = small_sky_catalog.compute()
    expected_columns = set(computed.columns) | {computed.index.name}
    assert expected_columns == lance_columns


def test_to_lance_overwrite(small_sky_catalog, tmp_path):
    """overwrite=True replaces an existing dataset."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)
    small_sky_catalog.to_lance(ds_path, overwrite=True)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_overwrite_false_raises(small_sky_catalog, tmp_path):
    """Writing to an existing dataset without overwrite=True raises an error."""
    pytest.importorskip("lancedb")
    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    with pytest.raises(ValueError):
        small_sky_catalog.to_lance(ds_path, overwrite=False)


def test_to_lance_data_matches(small_sky_catalog, tmp_path):
    """Values in the Lance dataset match the original catalog data."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    lance_df = tbl.to_pandas()

    original_df = small_sky_catalog.compute().reset_index()
    index_col = small_sky_catalog.compute().index.name

    lance_df = lance_df.sort_values(index_col).reset_index(drop=True)
    original_df = pd.DataFrame(original_df).sort_values(index_col).reset_index(drop=True)

    pd.testing.assert_frame_equal(lance_df, original_df, check_like=True)


def test_to_lance_multiple_partitions(small_sky_order1_catalog, tmp_path):
    """Catalogs with multiple partitions exercise the table.add() branch."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky_order1"
    assert len(small_sky_order1_catalog._ddf_pixel_map) > 1, "fixture must have >1 partition"
    small_sky_order1_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_order1_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_empty_catalog_raises(small_sky_catalog, tmp_path):
    """An all-empty catalog raises RuntimeError with an informative message."""
    pytest.importorskip("lancedb")
    ds_path = tmp_path / "small_sky"

    # Intentionally search an empty area so no partitions are iterated
    cone_search_catalog = small_sky_catalog.cone_search(0, -80, 1)

    with pytest.raises(RuntimeError, match="The output catalog is empty"):
        cone_search_catalog.to_lance(ds_path)


def test_to_lance_nested_partitions(small_sky_with_nested_sources, tmp_path):
    """Catalogs with nested sources has all rows included in the output dataset."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky_nested"
    assert len(small_sky_with_nested_sources._ddf_pixel_map) > 1, "fixture must have >1 partition"
    small_sky_with_nested_sources.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_with_nested_sources.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_data_matches_nested(small_sky_with_nested_sources, tmp_path):
    """Values in the Lance dataset match the original catalog data."""
    lancedb = pytest.importorskip("lancedb")

    ds_path = tmp_path / "small_sky_nested"
    small_sky_with_nested_sources.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    lance_df = tbl.to_pandas()

    original_df = small_sky_with_nested_sources.compute().reset_index()
    index_col = small_sky_with_nested_sources.compute().index.name

    lance_df = lance_df.sort_values(index_col).reset_index(drop=True)
    original_df = pd.DataFrame(original_df).sort_values(index_col).reset_index(drop=True)

    pd.testing.assert_frame_equal(lance_df, original_df, check_like=True)
