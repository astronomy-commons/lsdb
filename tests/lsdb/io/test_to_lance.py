import sys

import pandas as pd
import pytest

# `_map_s3_storage_options` and `_lance_storage_options_from_upath` are private helpers --
# they're intentionally not exported from `lsdb.io`. We import them directly from the
# submodule (rather than adding them to `lsdb/io/__init__.py`) so the public import
# structure of `lsdb.io` stays untouched.
from lsdb.io.to_lance import _lance_storage_options_from_upath, _map_s3_storage_options, to_lance


class _FakePath:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for a UPath, exposing only what `_lance_storage_options_from_upath` reads."""

    def __init__(self, protocol, storage_options=None):
        self.protocol = protocol
        self.storage_options = storage_options


def test_import_error_without_lancedb(monkeypatch):
    """Importing to_lance without lancedb installed raises a helpful ImportError."""
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
    assert small_sky_order1_catalog.npartitions > 1, "fixture must have >1 partition"
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
    assert small_sky_with_nested_sources.npartitions > 1, "fixture must have >1 partition"
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


# --- _map_s3_storage_options ------------------------------------------------
#
# These exercise the fsspec -> lance/object_store storage-options translation directly,
# with no network access, no `lancedb` install required, and no S3 emulator involved.


def test_map_s3_storage_options_all_fields_top_level():
    """Every field `_map_s3_storage_options` understands is mapped when given at the top level."""
    fsso = {
        "key": "AKIDEXAMPLE",
        "secret": "SECRETKEY",
        "token": "SESSIONTOKEN",
        "endpoint_url": "http://localhost:9000",
        "region_name": "us-east-1",
        "anon": False,
    }

    assert _map_s3_storage_options(fsso) == {
        "aws_endpoint": "http://localhost:9000",
        "allow_http": "true",  # http (not https) endpoint -> object_store needs an explicit opt-in
        "aws_region": "us-east-1",
        "aws_access_key_id": "AKIDEXAMPLE",
        "aws_secret_access_key": "SECRETKEY",
        "aws_session_token": "SESSIONTOKEN",
    }


def test_map_s3_storage_options_https_endpoint_omits_allow_http():
    """An https:// endpoint should not set allow_http."""
    lance_so = _map_s3_storage_options({"endpoint_url": "https://s3.amazonaws.com"})
    assert lance_so == {"aws_endpoint": "https://s3.amazonaws.com"}
    assert "allow_http" not in lance_so


def test_map_s3_storage_options_falls_back_to_client_kwargs():
    """endpoint_url/region_name nested under client_kwargs are read as a fallback."""
    fsso = {
        "client_kwargs": {
            "endpoint_url": "http://minio.local:9000",
            "region_name": "eu-west-1",
        }
    }

    assert _map_s3_storage_options(fsso) == {
        "aws_endpoint": "http://minio.local:9000",
        "allow_http": "true",
        "aws_region": "eu-west-1",
    }


def test_map_s3_storage_options_top_level_takes_precedence_over_client_kwargs():
    """Top-level endpoint_url/region_name win over the same keys in client_kwargs."""
    fsso = {
        "endpoint_url": "https://top-level.example.com",
        "region_name": "us-west-2",
        "client_kwargs": {
            "endpoint_url": "https://client-kwargs.example.com",
            "region_name": "eu-central-1",
        },
    }

    lance_so = _map_s3_storage_options(fsso)
    assert lance_so["aws_endpoint"] == "https://top-level.example.com"
    assert lance_so["aws_region"] == "us-west-2"


# --- _lance_storage_options_from_upath --------------------------------------


def test_lance_storage_options_from_upath_non_s3_returns_none():
    """Protocols other than s3 (local files, memory, http, ...) get no translation."""
    assert _lance_storage_options_from_upath(_FakePath(protocol="file")) is None
    assert _lance_storage_options_from_upath(_FakePath(protocol="memory")) is None


def test_lance_storage_options_from_upath_s3_delegates_to_mapper(monkeypatch):
    """For s3:// paths, the UPath's storage_options are handed to `_map_s3_storage_options`."""
    fsso = {"key": "AKID", "secret": "SECRET", "region_name": "us-east-2"}

    result = _lance_storage_options_from_upath(_FakePath(protocol="s3", storage_options=fsso))

    assert result == {
        "aws_access_key_id": "AKID",
        "aws_secret_access_key": "SECRET",
        "aws_region": "us-east-2",
    }


def test_lance_storage_options_from_upath_s3_handles_missing_storage_options():
    """An s3 UPath with no storage_options at all still resolves to an empty mapping."""
    assert not _lance_storage_options_from_upath(_FakePath(protocol="s3", storage_options=None))
