import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

# `_map_s3_storage_options` and `_lance_storage_options_from_upath` are private helpers --
# they're intentionally not exported from `lsdb.io`. We import them directly from the
# submodule (rather than adding them to `lsdb/io/__init__.py`) so the public import
# structure of `lsdb.io` stays untouched.
from lsdb.io.to_lance import _lance_storage_options_from_upath, _map_s3_storage_options, to_lance

# `lsdb/io/__init__.py` does `from .to_lance import to_lance`, which rebinds the `to_lance`
# attribute on the `lsdb.io` package to the function -- shadowing the submodule of the same
# name. `import lsdb.io.to_lance as to_lance_module` would therefore silently hand us that
# function instead of the module. Pull the real module out of sys.modules to be safe.
to_lance_module = sys.modules["lsdb.io.to_lance"]


class _FakePath:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for a UPath, exposing only what `_lance_storage_options_from_upath` reads."""

    def __init__(self, protocol, storage_options=None):
        self.protocol = protocol
        self.storage_options = storage_options


class _FakeS3Path:
    """Stand-in for a UPath pointing at s3:// that never touches the network.

    `to_lance` calls `.exists()` / `.iterdir()` (to check for a pre-existing dataset) and
    `/` (to build the `<table>.lance` subpath) before it ever gets to storage-options mapping.
    Faking those out lets us exercise the real `to_lance` -> `_lance_storage_options_from_upath`
    -> `_map_s3_storage_options` call chain without a live (or emulated) S3 endpoint.
    """

    def __init__(self, path, storage_options):
        self._path = str(path)
        self.protocol = "s3"
        self.storage_options = storage_options

    def __truediv__(self, other):
        return _FakeS3Path(f"{self._path}/{other}", self.storage_options)

    def __str__(self):
        return self._path

    def exists(self):
        return False

    def iterdir(self):
        return iter(())


class _FakeLanceTable:
    def add(self, *_args, **_kwargs):
        pass

    def optimize(self, *_args, **_kwargs):
        pass


class _FakeLanceDB:  # pylint: disable=too-few-public-methods
    def create_table(self, *_args, **_kwargs):
        return _FakeLanceTable()


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


def test_map_s3_storage_options_region_alias():
    """The bare `region` key is accepted when `region_name` is absent."""
    assert _map_s3_storage_options({"region": "ap-southeast-2"}) == {"aws_region": "ap-southeast-2"}


def test_map_s3_storage_options_anon():
    """anon=True maps to aws_skip_signature; credential fields (if any) are still mapped too."""
    lance_so = _map_s3_storage_options({"anon": True, "key": "irrelevant-but-still-mapped"})
    assert lance_so["aws_skip_signature"] == "true"
    assert lance_so["aws_access_key_id"] == "irrelevant-but-still-mapped"


def test_map_s3_storage_options_anon_false_omits_skip_signature():
    assert "aws_skip_signature" not in _map_s3_storage_options({"anon": False})


def test_map_s3_storage_options_empty_input():
    """No recognized fields -> empty output, not an error."""
    assert not _map_s3_storage_options({})


# --- _lance_storage_options_from_upath --------------------------------------


def test_lance_storage_options_from_upath_non_s3_returns_none():
    """Protocols other than s3 (local files, memory, http, ...) get no translation."""
    assert _lance_storage_options_from_upath(_FakePath(protocol="file")) is None
    assert _lance_storage_options_from_upath(_FakePath(protocol="memory")) is None


def test_lance_storage_options_from_upath_s3_delegates_to_mapper(monkeypatch):
    """For s3:// paths, the UPath's storage_options are handed to `_map_s3_storage_options`."""
    fsso = {"key": "AKID", "secret": "SECRET", "region_name": "us-east-2"}
    spy = MagicMock(wraps=_map_s3_storage_options)
    monkeypatch.setattr(to_lance_module, "_map_s3_storage_options", spy)

    result = _lance_storage_options_from_upath(_FakePath(protocol="s3", storage_options=fsso))

    spy.assert_called_once_with(fsso)
    assert result == {
        "aws_access_key_id": "AKID",
        "aws_secret_access_key": "SECRET",
        "aws_region": "us-east-2",
    }


def test_lance_storage_options_from_upath_s3_handles_missing_storage_options():
    """An s3 UPath with no storage_options at all still resolves to an empty mapping."""
    assert not _lance_storage_options_from_upath(_FakePath(protocol="s3", storage_options=None))


# --- Wiring: to_lance -> _lance_storage_options_from_upath -> _map_s3_storage_options ----


def test_to_lance_passes_mapped_s3_storage_options_to_lancedb_connect(monkeypatch, small_sky_catalog):
    """`to_lance` maps a catalog path's S3 storage_options and forwards the result to
    `lancedb.connect`, without ever touching a real (or emulated) S3 endpoint.
    """
    fsso = {
        "key": "AKIDEXAMPLE",
        "secret": "SECRETKEY",
        "token": "SESSIONTOKEN",
        "endpoint_url": "http://localhost:9000",
        "region_name": "us-east-1",
    }
    expected_lance_so = {
        "aws_endpoint": "http://localhost:9000",
        "allow_http": "true",
        "aws_region": "us-east-1",
        "aws_access_key_id": "AKIDEXAMPLE",
        "aws_secret_access_key": "SECRETKEY",
        "aws_session_token": "SESSIONTOKEN",
    }

    spy = MagicMock(wraps=_map_s3_storage_options)
    monkeypatch.setattr(to_lance_module, "_map_s3_storage_options", spy)
    monkeypatch.setattr(to_lance_module, "UPath", lambda p, *a, **kw: _FakeS3Path(p, fsso))  # noqa: ARG005

    connect_calls = []

    def fake_connect(path, storage_options=None):
        connect_calls.append({"path": path, "storage_options": storage_options})
        return _FakeLanceDB()

    fake_lancedb = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)

    to_lance(small_sky_catalog, base_catalog_path="s3://some-bucket/small_sky", optimize_dataset=False)

    spy.assert_called_once_with(fsso)
    assert len(connect_calls) == 1
    assert connect_calls[0]["storage_options"] == expected_lance_so
