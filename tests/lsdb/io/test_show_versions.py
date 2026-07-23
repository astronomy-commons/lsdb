import shutil
from types import SimpleNamespace

import lsdb
from lsdb.io.show_versions import _format_catalog_versions


def test_show_versions(capsys):
    lsdb.show_versions()
    captured = capsys.readouterr().out
    assert captured.startswith("\n--------      SYSTEM INFO      --------")
    assert "lsdb" in captured
    assert "hats" in captured
    assert "nested-pandas" in captured
    assert "pyarrow" in captured
    assert "fsspec" in captured


def test_collection_version_present(small_sky_order1_collection_dir, tmp_path):
    """The accessor returns the value for a collection that sets the key."""
    collection_dir = tmp_path / "versioned_collection"
    shutil.copytree(str(small_sky_order1_collection_dir), str(collection_dir))
    props = collection_dir / "collection.properties"
    props.write_text(props.read_text() + "\ncollection_version=v3.0.0\n")

    catalog = lsdb.open_catalog(collection_dir)
    assert catalog.collection_version == "v3.0.0"


def test_collection_version_absent_collection(small_sky_order1_collection_dir):
    """A collection without the key reports None."""
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir)
    assert catalog.collection_version is None


def test_collection_version_standalone(small_sky_catalog):
    """A standalone catalog (no collection) reports None."""
    assert small_sky_catalog.hc_collection is None
    assert small_sky_catalog.collection_version is None


# --- Stand-in builders for the display tests ---
# show_catalog_versions is duck-typed: it only reads ``name`` and ``hc_collection``, so
# lightweight namespaces stand in for real catalogs.


def _collection(name, version):
    return SimpleNamespace(collection_properties=SimpleNamespace(name=name), collection_version=version)


def _catalog(name, collection=None):
    return SimpleNamespace(name=name, hc_collection=collection)


def test_show_catalog_versions_version_set():
    obj = _catalog("gaia_dr3_object", _collection("gaia_dr3", "v2.1.0"))
    assert _format_catalog_versions(obj) == "Collection: gaia_dr3   v2.1.0"


def test_show_catalog_versions_version_unset():
    obj = _catalog("gaia_dr3_object", _collection("gaia_dr3", None))
    assert _format_catalog_versions(obj) == "Collection: gaia_dr3   —"


def test_show_catalog_versions_standalone():
    obj = _catalog("small_sky")
    assert _format_catalog_versions(obj) == "small_sky   (standalone catalog, no collection version)"


def test_show_catalog_versions_dedupes_shared_collection():
    collection = _collection("gaia_dr3", "v2.1.0")
    obj = _catalog("gaia_dr3_object", collection)
    source = _catalog("gaia_dr3_source", collection)
    assert _format_catalog_versions(obj, source) == "Collection: gaia_dr3   v2.1.0"


def test_show_catalog_versions_multiple_aligned():
    gaia = _catalog("gaia_obj", _collection("gaia_dr3", "v2.1.0"))
    ztf = _catalog("ztf_obj", _collection("ztf", "v20"))
    standalone = _catalog("small_sky")
    assert _format_catalog_versions(gaia, ztf, standalone) == (
        "Collection: gaia_dr3   v2.1.0\n"
        "Collection: ztf        v20\n"
        "small_sky              (standalone catalog, no collection version)"
    )


def test_show_catalog_versions_no_args():
    assert _format_catalog_versions() == "No catalogs provided."


def test_show_catalog_versions_prints(capsys, small_sky_order1_collection_dir):
    """The public function prints the tree for a real (versionless) collection."""
    catalog = lsdb.open_catalog(small_sky_order1_collection_dir)
    lsdb.show_catalog_versions(catalog)
    captured = capsys.readouterr().out
    assert "Collection: small_sky_order1_collection" in captured
    # A versionless collection renders with an em dash.
    assert "—" in captured
