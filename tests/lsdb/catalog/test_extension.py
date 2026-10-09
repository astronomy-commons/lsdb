import shutil

import hats
import pandas as pd
import pytest
from hats.catalog import ExtensionProperties

import lsdb

EXTENSION_NAME = "small_sky_order1_lightcurves"


@pytest.fixture
def small_sky_with_sources_extension(small_sky_with_sources_extension_dir):
    return lsdb.open_catalog(small_sky_with_sources_extension_dir)


def copy_collection(collection_dir, tmp_path, *, update_extension=None, properties=None):
    """Copy the collection, rewriting the extension data with `update_extension(extension)`,
    and updating the extension `properties`."""
    collection = tmp_path / "collection"
    shutil.copytree(collection_dir, collection)
    if update_extension is not None:
        shutil.rmtree(collection / EXTENSION_NAME)
        extension = lsdb.open_catalog(collection_dir / EXTENSION_NAME, columns="all")
        update_extension(extension).write_catalog(
            collection / EXTENSION_NAME, catalog_name=EXTENSION_NAME, progress_bar=False
        )
    if properties is not None:
        extension_properties = ExtensionProperties.read_from_file(collection / f"{EXTENSION_NAME}.properties")
        extension_properties.model_copy(update=properties).to_properties_file(collection)
    return collection


def assert_same_rows(result, expected):
    """The rows of the extended catalog are the rows of the catalog it was split from."""
    result = result.sort_values("id")
    pd.testing.assert_frame_equal(result, expected.sort_values("id")[result.columns])


def test_load_extension(small_sky_with_sources_extension, small_sky_with_nested_sources):
    catalog = small_sky_with_sources_extension.load_extension(EXTENSION_NAME)
    assert catalog.columns.tolist() == [*small_sky_with_sources_extension.columns, "sources"]
    assert catalog.get_healpix_pixels() == small_sky_with_sources_extension.get_healpix_pixels()
    assert catalog.margin is None
    assert catalog.all_extensions == [EXTENSION_NAME]
    assert catalog.extensions == (EXTENSION_NAME,)
    assert_same_rows(catalog.compute(), small_sky_with_nested_sources.compute())


def test_all_extensions(small_sky_with_sources_extension, small_sky_order1_dir):
    assert small_sky_with_sources_extension.all_extensions == [EXTENSION_NAME]
    # The extensions of the collection are still listed after operations on the catalog.
    assert small_sky_with_sources_extension.query("id > 750").all_extensions == [EXTENSION_NAME]
    assert small_sky_with_sources_extension[["id", "ra", "dec"]].all_extensions == [EXTENSION_NAME]
    assert small_sky_with_sources_extension.cone_search(0, -80, 20 * 3600).all_extensions == [EXTENSION_NAME]
    # A stand-alone catalog has no extensions.
    assert lsdb.open_catalog(small_sky_order1_dir).all_extensions == []


def test_load_extension_nested_subcolumns(small_sky_with_sources_extension, small_sky_with_nested_sources):
    catalog = small_sky_with_sources_extension.load_extension(
        EXTENSION_NAME, columns=["sources.mjd", "sources.mag"]
    )
    assert catalog.columns.tolist() == [*small_sky_with_sources_extension.columns, "sources"]
    assert catalog.meta["sources"].nest.columns == ["mjd", "mag"]
    expected = small_sky_with_nested_sources[
        ["id", "ra", "dec", "ra_error", "dec_error", "sources.mjd", "sources.mag"]
    ]
    assert_same_rows(catalog.compute(), expected.compute())


def test_load_extension_after_search(small_sky_with_sources_extension, small_sky_with_nested_sources):
    search = lsdb.ConeSearch(ra=0, dec=-80, radius_arcsec=15 * 3600)
    catalog = small_sky_with_sources_extension.search(search).load_extension(EXTENSION_NAME)
    assert_same_rows(catalog.compute(), small_sky_with_nested_sources.search(search).compute())


def test_extensions(small_sky_with_sources_extension):
    assert small_sky_with_sources_extension.extensions == ()
    catalog = small_sky_with_sources_extension.load_extension(EXTENSION_NAME)
    assert catalog.extensions == (EXTENSION_NAME,)
    # Loading an extension does not change the catalog it was loaded on.
    assert small_sky_with_sources_extension.extensions == ()
    # The loaded extensions are still listed after operations on the catalog.
    assert catalog.query("id > 750").extensions == (EXTENSION_NAME,)
    assert catalog.cone_search(0, -80, 20 * 3600).extensions == (EXTENSION_NAME,)
    assert catalog[["id", "sources"]].extensions == (EXTENSION_NAME,)
    with pytest.raises(ValueError, match=f"`{EXTENSION_NAME}` is already loaded"):
        catalog.load_extension(EXTENSION_NAME)


def test_load_extension_from_path(small_sky_with_sources_extension_dir, small_sky_with_nested_sources):
    """An extension that is not listed in a collection, loaded under a name of our choice."""
    catalog = lsdb.open_catalog(small_sky_with_sources_extension_dir / "small_sky_order1_nested_sources")
    assert catalog.all_extensions == []
    path = small_sky_with_sources_extension_dir / f"{EXTENSION_NAME}.properties"
    catalog = catalog.load_extension("lightcurves", path=path)
    assert catalog.extensions == ("lightcurves",)
    assert_same_rows(catalog.compute(), small_sky_with_nested_sources.compute())


def test_load_extension_errors(
    small_sky_with_sources_extension, small_sky_order1_dir, small_sky_with_sources_extension_dir
):
    with pytest.raises(ValueError, match="`other_extension` is not specified in all_extensions"):
        small_sky_with_sources_extension.load_extension("other_extension")
    with pytest.raises(ValueError, match="not part of a collection, so give the `path`"):
        lsdb.open_catalog(small_sky_order1_dir).load_extension(EXTENSION_NAME)
    with pytest.raises(ValueError, match="is not a HATS extension"):
        small_sky_with_sources_extension.load_extension(
            "core", path=small_sky_with_sources_extension_dir / "small_sky_order1_nested_sources"
        )


def test_load_extension_without_listed_columns(small_sky_with_sources_extension_dir, tmp_path):
    collection = copy_collection(
        small_sky_with_sources_extension_dir, tmp_path, properties={"extension_columns": None}
    )
    catalog = lsdb.open_catalog(collection)
    with pytest.raises(ValueError, match="does not list its columns"):
        catalog.load_extension(EXTENSION_NAME)
    assert catalog.load_extension(EXTENSION_NAME, columns=["sources"]).columns[-1] == "sources"


@pytest.mark.parametrize("join_style", ["left", "inner"])
def test_load_extension_join_style(
    small_sky_with_sources_extension_dir, tmp_path, small_sky_with_nested_sources, join_style
):
    """Only some of the rows have extension data."""
    collection = copy_collection(
        small_sky_with_sources_extension_dir,
        tmp_path,
        update_extension=lambda extension: extension.query("object_id < 750"),
        properties={"extension_join_style": join_style},
    )
    result = lsdb.open_catalog(collection).load_extension(EXTENSION_NAME).compute()
    expected = small_sky_with_nested_sources.compute()
    has_extension = result["id"] < 750
    assert_same_rows(result[has_extension], expected[expected["id"] < 750])
    if join_style == "left":
        assert len(result) == len(expected)
        assert result.loc[~has_extension, "sources"].isna().all()
    else:
        assert has_extension.all()


def test_load_extension_same_join_column_name(
    small_sky_with_sources_extension_dir, tmp_path, small_sky_with_nested_sources
):
    """The extension joins on a column with the same name as the catalog's, which is the default
    of hats-import. Only the catalog's is kept."""
    collection = copy_collection(
        small_sky_with_sources_extension_dir,
        tmp_path,
        update_extension=lambda extension: extension.rename({"object_id": "id"}),
        properties={"join_column": "id"},
    )
    catalog = lsdb.open_catalog(collection)
    extended = catalog.load_extension(EXTENSION_NAME)
    assert extended.columns.tolist() == [*catalog.columns, "sources"]
    assert_same_rows(extended.compute(), small_sky_with_nested_sources.compute())


def test_load_extension_storage_options(small_sky_with_sources_extension, small_sky_order1_dir, monkeypatch):
    """The storage options given to `load_extension` override those of the catalog's collection."""
    read_storage_options = []
    read_hats = hats.read_hats

    def recording_read_hats(path, storage_options=None, **kwargs):
        if str(path).endswith(".properties"):
            read_storage_options.append(storage_options)
        return read_hats(path, **kwargs)

    monkeypatch.setattr(hats, "read_hats", recording_read_hats)
    small_sky_with_sources_extension.hc_collection.storage_options = {"collection": "option"}
    small_sky_with_sources_extension.load_extension(EXTENSION_NAME)
    small_sky_with_sources_extension.load_extension(EXTENSION_NAME, storage_options={"given": "option"})
    small_sky_with_sources_extension.load_extension(EXTENSION_NAME, storage_options={})
    extension_path = small_sky_with_sources_extension.hc_collection.get_extension_path(EXTENSION_NAME)
    lsdb.open_catalog(small_sky_order1_dir).load_extension("lc", path=extension_path)
    assert read_storage_options == [{"collection": "option"}, {"given": "option"}, {}, None]
