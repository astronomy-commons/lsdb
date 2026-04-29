import importlib
import sys

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")


def test_import_error_without_lancedb(monkeypatch):
    """Importing to_lance without lancedb installed raises a helpful ImportError."""
    monkeypatch.setitem(sys.modules, "lancedb", None)
    monkeypatch.delitem(sys.modules, "lsdb.io.to_lance", raising=False)

    with pytest.raises(ImportError, match="to_lance requires the `lancedb` package"):
        importlib.import_module("lsdb.io.to_lance")


def test_to_lance_writes_dataset(small_sky_catalog, tmp_path):
    """Basic write: dataset exists and row count matches."""
    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_columns_match(small_sky_catalog, tmp_path):
    """All catalog columns (including the spatial index) appear in the Lance dataset."""
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
    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)
    small_sky_catalog.to_lance(ds_path, overwrite=True)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_overwrite_false_raises(small_sky_catalog, tmp_path):
    """Writing to an existing dataset without overwrite=True raises an error."""
    ds_path = tmp_path / "small_sky"
    small_sky_catalog.to_lance(ds_path)

    with pytest.raises(ValueError):
        small_sky_catalog.to_lance(ds_path, overwrite=False)


def test_to_lance_data_matches(small_sky_catalog, tmp_path):
    """Values in the Lance dataset match the original catalog data."""
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
    ds_path = tmp_path / "small_sky_order1"
    assert len(small_sky_order1_catalog._ddf_pixel_map) > 1, "fixture must have >1 partition"
    small_sky_order1_catalog.to_lance(ds_path)

    db = lancedb.connect(str(ds_path))
    tbl = db.open_table("data")
    expected_rows = len(small_sky_order1_catalog.compute())
    assert tbl.count_rows() == expected_rows


def test_to_lance_empty_catalog_raises(small_sky_catalog, tmp_path, monkeypatch):
    """An all-empty catalog raises RuntimeError with an informative message."""
    ds_path = tmp_path / "small_sky"

    # Patch the pixel map to be empty so no partitions are iterated
    monkeypatch.setattr(small_sky_catalog, "_ddf_pixel_map", {})

    with pytest.raises(RuntimeError, match="The output catalog is empty"):
        small_sky_catalog.to_lance(ds_path)
