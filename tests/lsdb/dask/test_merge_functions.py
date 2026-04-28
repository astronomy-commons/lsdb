import logging

import nested_pandas as npd
import pandas as pd
from hats.pixel_math import HealpixPixel

from lsdb.operations.functions.merge_catalog_functions import (
    align_catalog_to_partitions,
    create_merged_catalog_info,
    create_pixel_ndfs,
    perform_align_and_apply_func,
)


def test_create_merged_catalog_info_suffix_logging(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    with caplog.at_level(logging.WARNING):
        merged_catalog_info = create_merged_catalog_info(
            small_sky_catalog,
            small_sky_xmatch_catalog,
            "merged_catalog",
            ("_left", "_right"),
            suffix_method="overlapping_columns",
        )
        assert merged_catalog_info.catalog_name == "merged_catalog"
        assert merged_catalog_info.ra_column == "ra_left"
        assert merged_catalog_info.dec_column == "dec_left"
        assert "Renaming overlapping columns" not in caplog.text


def test_align_catalog_to_partitions_none_catalog(small_sky_order1_catalog):
    """When catalog is None, each partition should be an empty 0-column DataFrame."""
    pixels = small_sky_order1_catalog.get_healpix_pixels()
    result = align_catalog_to_partitions(None, pixels)
    assert result.npartitions == len(pixels)
    for i in range(result.npartitions):
        part = result.partitions[i].compute()
        assert len(part.columns) == 0
        assert len(part) == 0


def test_align_catalog_to_partitions_in_tree(small_sky_order1_catalog):
    """Pixels present in the catalog's pixel tree should return non-empty partitions with catalog schema."""
    pixels = small_sky_order1_catalog.get_healpix_pixels()
    result = align_catalog_to_partitions(small_sky_order1_catalog, pixels)
    assert result.npartitions == len(pixels)
    for i, p in enumerate(pixels):
        part = result.partitions[i].compute()
        pd.testing.assert_frame_equal(
            part, small_sky_order1_catalog.get_partition(p.order, p.pixel).compute()
        )


def test_align_catalog_to_partitions_pixel_not_in_tree(small_sky_order1_catalog):
    """Pixels absent from the catalog's pixel tree should come back as empty (0-row) partitions
    that still carry the catalog's column schema (not 0-column frames)."""
    pixels = [HealpixPixel(0, 100)]
    result = align_catalog_to_partitions(small_sky_order1_catalog, pixels)
    assert result.npartitions == 1
    part = result.get_partition(0).compute()
    assert len(part) == 0
    assert all(part.columns == small_sky_order1_catalog._ddf.columns)


def test_align_catalog_to_partitions_none_pixel(small_sky_order1_catalog):
    """A None entry in the pixel list should behave the same as a missing pixel."""
    result = align_catalog_to_partitions(small_sky_order1_catalog, [None])
    assert result.npartitions == 1
    part = result.get_partition(0).compute()
    assert len(part) == 0
    assert all(part.columns == small_sky_order1_catalog._ddf.columns)


def test_create_pixel_ndfs_real_pixels():
    pixels = [HealpixPixel(1, 0), HealpixPixel(2, 3)]
    result = create_pixel_ndfs(pixels)
    assert result.npartitions == len(pixels)
    p0 = result.get_partition(0).compute()
    assert p0.iloc[0]["order"] == 1
    assert p0.iloc[0]["pixel"] == 0
    p1 = result.get_partition(1).compute()
    assert p1.iloc[0]["order"] == 2
    assert p1.iloc[0]["pixel"] == 3


def test_create_pixel_ndfs_none_pixels():
    """None pixels should be encoded with the -99 sentinel."""
    pixels = [None, HealpixPixel(1, 0), None]
    result = create_pixel_ndfs(pixels)
    assert result.npartitions == 3
    sentinel_part = result.get_partition(0).compute()
    assert sentinel_part.iloc[0]["order"] == -99
    assert sentinel_part.iloc[0]["pixel"] == -99
    real_part = result.get_partition(1).compute()
    assert real_part.iloc[0]["order"] == 1


def _make_pixel_df(order, pixel):
    """Helper: build the single-row pixel DataFrame that perform_align_and_apply_func expects."""
    return pd.DataFrame({"order": [order], "pixel": [pixel]})


def test_perform_align_and_apply_func_passes_dataframes():
    """Non-empty partitions should be forwarded to the inner function after stripping hips columns."""
    df = npd.NestedFrame({"ra": [1.0], "dec": [2.0]})
    pixel_df = _make_pixel_df(1, 0)

    received = {}

    def func(part, pix):
        received["part"] = part
        received["pix"] = pix
        return part

    perform_align_and_apply_func(1, func, df, pixel_df)

    assert received["pix"] == HealpixPixel(1, 0)
    assert list(received["part"].columns) == ["ra", "dec"]


def test_perform_align_and_apply_func_empty_df_becomes_none():
    """A 0-column DataFrame (stand-in for a missing/None catalog) must arrive as None."""
    empty_df = pd.DataFrame()  # 0 columns
    pixel_df = _make_pixel_df(1, 0)

    received = {}

    def func(part, pix):
        received["part"] = part
        received["pix"] = pix
        return npd.NestedFrame()

    perform_align_and_apply_func(1, func, empty_df, pixel_df)

    assert received["part"] is None


def test_perform_align_and_apply_func_none_pixel_sentinel():
    """The -99 sentinel in the pixel DataFrame should decode back to None."""
    df = npd.NestedFrame({"ra": [1.0]})
    pixel_df = _make_pixel_df(-99, -99)

    received = {}

    def func(part, pix):
        received["pix"] = pix
        return part

    perform_align_and_apply_func(1, func, df, pixel_df)

    assert received["pix"] is None
