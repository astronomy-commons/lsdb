from pathlib import Path

import pandas as pd
import pytest

import lsdb

lance = pytest.importorskip("lance")


def test_to_lance_writes_dataset(small_sky_catalog, tmp_path):
    """Basic write: dataset exists and row count matches."""
    ds_path = tmp_path / "small_sky.lance"
    small_sky_catalog.to_lance(ds_path)

    ds = lance.dataset(str(ds_path))
    expected_rows = len(small_sky_catalog.compute())
    assert ds.count_rows() == expected_rows


def test_to_lance_columns_match(small_sky_catalog, tmp_path):
    """All catalog columns (including the spatial index) appear in the Lance dataset."""
    ds_path = tmp_path / "small_sky.lance"
    small_sky_catalog.to_lance(ds_path)

    ds = lance.dataset(str(ds_path))
    lance_columns = set(ds.schema.names)

    computed = small_sky_catalog.compute()
    expected_columns = set(computed.columns) | {computed.index.name}
    assert expected_columns == lance_columns


def test_to_lance_overwrite(small_sky_catalog, tmp_path):
    """overwrite=True replaces an existing dataset."""
    ds_path = tmp_path / "small_sky.lance"
    small_sky_catalog.to_lance(ds_path)
    small_sky_catalog.to_lance(ds_path, overwrite=True)

    ds = lance.dataset(str(ds_path))
    expected_rows = len(small_sky_catalog.compute())
    assert ds.count_rows() == expected_rows


def test_to_lance_overwrite_false_raises(small_sky_catalog, tmp_path):
    """Writing to an existing dataset without overwrite=True raises an error."""
    ds_path = tmp_path / "small_sky.lance"
    small_sky_catalog.to_lance(ds_path)

    with pytest.raises(Exception):
        small_sky_catalog.to_lance(ds_path, overwrite=False)


def test_to_lance_data_matches(small_sky_catalog, tmp_path):
    """Values in the Lance dataset match the original catalog data."""
    ds_path = tmp_path / "small_sky.lance"
    small_sky_catalog.to_lance(ds_path)

    ds = lance.dataset(str(ds_path))
    lance_df = ds.to_table().to_pandas()

    original_df = small_sky_catalog.compute().reset_index()
    index_col = small_sky_catalog.compute().index.name

    lance_df = lance_df.sort_values(index_col).reset_index(drop=True)
    original_df = original_df.sort_values(index_col).reset_index(drop=True)

    pd.testing.assert_frame_equal(lance_df, original_df, check_like=True)
