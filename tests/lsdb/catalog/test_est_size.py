import logging

import pyarrow as pa
import pytest
from hats.catalog.catalog_snapshot import CatalogSnapshot

import lsdb
from lsdb.catalog.dataset.healpix_dataset import COMPUTE_SIZE_WARNING_THRESHOLD_KiB
from lsdb.io.schema import get_arrow_schema


def _row_bytes(schema):
    """Sum the fixed-width byte sizes of all fields in a schema."""
    return sum(f.type.bit_width // 8 for f in schema)


def test_est_size_full_columns_equals_hats_estsize(small_sky_order1_catalog):
    # All columns loaded: column_ratio = 1, partition_ratio = 1
    hats_estsize = small_sky_order1_catalog.hc_structure.catalog_info.hats_estsize
    assert small_sky_order1_catalog.est_size() == pytest.approx(float(hats_estsize))


def test_est_size_default_column_subset(small_sky_order1_default_cols_catalog):
    cat = small_sky_order1_default_cols_catalog
    hats_estsize = cat.hc_structure.catalog_info.hats_estsize
    original_schema = cat.hc_structure.original_schema
    current_schema = get_arrow_schema(cat._ddf)
    expected = hats_estsize * (_row_bytes(current_schema) / _row_bytes(original_schema))
    assert cat.est_size() == pytest.approx(expected)


def test_est_size_explicit_two_column_subset(small_sky_order1_default_cols_catalog):
    cat = small_sky_order1_default_cols_catalog[["ra", "dec"]]
    hats_estsize = cat.hc_structure.catalog_info.hats_estsize
    original_schema = cat.hc_structure.original_schema
    current_schema = get_arrow_schema(cat._ddf)
    expected = hats_estsize * (_row_bytes(current_schema) / _row_bytes(original_schema))
    assert cat.est_size() == pytest.approx(expected)


def test_est_size_scales_with_partition_subset(small_sky_order1_catalog):
    cat = small_sky_order1_catalog
    hats_estsize = cat.hc_structure.catalog_info.hats_estsize
    original_n_partitions = len(cat.hc_structure.snapshot.partition_info)
    pixel = cat.get_healpix_pixels()[0]
    subset = cat.search(lsdb.PixelSearch([pixel]))
    assert len(subset.get_healpix_pixels()) == 1
    expected = float(hats_estsize) * (1 / original_n_partitions)
    assert subset.est_size() == pytest.approx(expected)


def test_est_size_column_and_partition_ratios_combine(small_sky_order1_default_cols_catalog):
    cat = small_sky_order1_default_cols_catalog
    hats_estsize = cat.hc_structure.catalog_info.hats_estsize
    original_schema = cat.hc_structure.original_schema
    original_n_partitions = len(cat.hc_structure.snapshot.partition_info)
    pixel = cat.get_healpix_pixels()[0]
    subset = cat[["ra", "dec"]].search(lsdb.PixelSearch([pixel]))
    current_schema = get_arrow_schema(subset._ddf)
    expected = (
        hats_estsize
        * (1 / original_n_partitions)
        * (_row_bytes(current_schema) / _row_bytes(original_schema))
    )
    assert subset.est_size() == pytest.approx(expected)


def test_est_size_nested(small_sky_with_nested_sources):
    cat = small_sky_with_nested_sources[["sources"]]
    assert cat.est_size() < small_sky_with_nested_sources.est_size()
    assert cat.est_size() > small_sky_with_nested_sources.est_size() * 0.5


def test_compute_warns_when_estimated_size_exceeds_threshold(small_sky_order1_catalog, mocker, caplog):
    mocker.patch.object(
        small_sky_order1_catalog, "est_size", return_value=COMPUTE_SIZE_WARNING_THRESHOLD_KB + 1
    )
    with caplog.at_level(logging.WARNING):
        small_sky_order1_catalog.compute(progress_bar=False)
    assert "estimated size" in caplog.text.lower()


def test_compute_does_not_warn_below_threshold(small_sky_order1_catalog, mocker, caplog):
    mocker.patch.object(
        small_sky_order1_catalog, "est_size", return_value=COMPUTE_SIZE_WARNING_THRESHOLD_KB - 1
    )
    with caplog.at_level(logging.WARNING):
        small_sky_order1_catalog.compute(progress_bar=False)
    assert "estimated size" not in caplog.text.lower()


def test_repr_html_includes_estimated_size(small_sky_order1_catalog):
    html = small_sky_order1_catalog._repr_html_()
    est_size = small_sky_order1_catalog.est_size()
    assert "estimated size" in html.lower()
    assert f"{est_size:.1f} KB" in html


def test_repr_html_omits_size_without_snapshot(small_sky_order1_catalog):
    small_sky_order1_catalog.hc_structure.snapshot = None
    html = small_sky_order1_catalog._repr_html_()
    assert "estimated size" not in html.lower()


def test_est_size_none_without_snapshot(small_sky_order1_catalog):
    small_sky_order1_catalog.hc_structure.snapshot = None
    assert small_sky_order1_catalog.est_size() is None


def test_est_size_none_without_estsize(small_sky_order1_catalog):
    small_sky_order1_catalog.hc_structure.catalog_info.hats_estsize = None
    assert small_sky_order1_catalog.est_size() is None


def test_est_size_none_when_column_not_in_original_schema(small_sky_order1_default_cols_catalog, mocker):
    extra_schema = pa.schema([pa.field("ra", pa.float64()), pa.field("derived_col", pa.int64())])
    mocker.patch("lsdb.catalog.dataset.healpix_dataset.get_arrow_schema", return_value=extra_schema)
    assert small_sky_order1_default_cols_catalog.est_size() is None


def test_est_size_none_for_variable_width_original_schema_without_total_rows(
    small_sky_order1_default_cols_catalog,
):

    string_schema = pa.schema([pa.field("name", pa.string()), pa.field("ra", pa.float64())])
    original_snapshot = small_sky_order1_default_cols_catalog.hc_structure.snapshot
    new_catalog_info = original_snapshot.catalog_info.copy_and_update(total_rows=None)
    small_sky_order1_default_cols_catalog.hc_structure.snapshot = CatalogSnapshot(
        schema=string_schema,
        catalog_info=new_catalog_info,
        partition_info=original_snapshot.partition_info,
    )
    assert small_sky_order1_default_cols_catalog.est_size() is None
