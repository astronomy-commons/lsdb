import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb")


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


def test_to_lance_empty_catalog_raises(small_sky_order1_catalog, tmp_path):
    """An all-empty catalog raises RuntimeError with an informative message."""
    ds_path = tmp_path / "small_sky"

    # cone_search(0, -80, 1) is known to produce an empty result for this catalog
    empty_catalog = small_sky_order1_catalog.cone_search(0, -80, 1)
    assert empty_catalog._ddf.npartitions >= 1

    non_empty_pixels = [
        pixel
        for pixel, partition_index in empty_catalog._ddf_pixel_map.items()
        if len(empty_catalog._ddf.partitions[partition_index]) > 0
    ]
    assert len(non_empty_pixels) == 0

    with pytest.raises(RuntimeError, match="The output catalog is empty"):
        empty_catalog.to_lance(ds_path)
