import shutil
from pathlib import Path

import pandas as pd
import pytest
from hats.catalog import CollectionProperties, ExtensionProperties
from upath import UPath

import lsdb


def test_load_extension(small_sky_with_sources_extension, small_sky_with_nested_sources):
    catalog = small_sky_with_sources_extension.load_extension("lightcurves")
    assert catalog.columns.tolist() == [*small_sky_with_sources_extension.columns, "sources"]
    assert catalog.get_healpix_pixels() == small_sky_with_sources_extension.get_healpix_pixels()
    assert catalog.extensions == ("lightcurves",)
    assert catalog.all_extensions == {"lightcurves": "small_sky_order1_lightcurves.properties"}
    assert catalog.margin is None
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_with_nested_sources.compute())


def test_all_extensions(small_sky_with_sources_extension, small_sky_order1_dir):
    all_extensions = {"lightcurves": "small_sky_order1_lightcurves.properties"}
    assert small_sky_with_sources_extension.all_extensions == all_extensions
    # The extensions of the collection are still listed after operations on the catalog.
    assert small_sky_with_sources_extension.query("id > 750").all_extensions == all_extensions
    assert small_sky_with_sources_extension[["id", "ra", "dec"]].all_extensions == all_extensions
    assert small_sky_with_sources_extension.cone_search(0, -80, 20 * 3600).all_extensions == all_extensions
    # A standalone catalog has no extensions.
    assert lsdb.open_catalog(small_sky_order1_dir).all_extensions == {}


def test_extensions(small_sky_with_sources_extension):
    assert small_sky_with_sources_extension.extensions == ()
    catalog = small_sky_with_sources_extension.load_extension("lightcurves")
    assert catalog.extensions == ("lightcurves",)
    # The loaded extensions are still listed after operations on the catalog.
    assert catalog.query("id > 750").extensions == ("lightcurves",)
    assert catalog.cone_search(0, -80, 20 * 3600).extensions == ("lightcurves",)
    assert catalog[["id", "sources"]].extensions == ("lightcurves",)


def test_load_extension_with_columns(small_sky_with_sources_extension, small_sky_with_nested_sources):
    catalog = small_sky_with_sources_extension.load_extension(
        "lightcurves", columns=["sources.mjd", "sources.mag"]
    )
    assert catalog.columns.tolist() == [*small_sky_with_sources_extension.columns, "sources"]
    assert catalog.meta["sources"].columns == ["mjd", "mag"]
    expected = small_sky_with_nested_sources[
        ["id", "ra", "dec", "ra_error", "dec_error", "sources.mjd", "sources.mag"]
    ]
    pd.testing.assert_frame_equal(catalog.compute(), expected.compute())


def test_load_extension_after_search(small_sky_with_sources_extension, small_sky_with_nested_sources):
    search = lsdb.ConeSearch(ra=0, dec=-80, radius_arcsec=15 * 3600)
    catalog = small_sky_with_sources_extension.search(search).load_extension("lightcurves")
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_with_nested_sources.search(search).compute())


@pytest.mark.parametrize("path_type", [str, Path, UPath])
def test_load_extension_from_path(
    small_sky_with_sources_extension_dir, small_sky_with_nested_sources, path_type
):
    catalog = lsdb.open_catalog(small_sky_with_sources_extension_dir / "small_sky_order1_nested_sources")
    assert catalog.all_extensions == {}
    properties_path = small_sky_with_sources_extension_dir / "small_sky_order1_lightcurves.properties"
    catalog = catalog.load_extension(path_type(properties_path))
    # The extensions name is read from the extension properties file.
    assert catalog.extensions == ("small_sky_order1_lightcurves",)
    pd.testing.assert_frame_equal(catalog.compute(), small_sky_with_nested_sources.compute())


def test_load_extension_inner_join(
    small_sky_with_sources_extension_dir, tmp_path, small_sky_with_nested_sources
):
    """Only some of the rows have extension data. The others are dropped."""
    collection = copy_collection(
        small_sky_with_sources_extension_dir,
        tmp_path,
        update_extension=lambda extension: extension.query("object_id < 750"),
        properties={"extension_join_style": "inner"},
    )
    result = lsdb.open_catalog(collection).load_extension("lightcurves").compute()
    extension = lsdb.open_catalog(collection / "small_sky_order1_lightcurves")
    assert len(result) == len(extension) == 50
    expected = small_sky_with_nested_sources.compute()
    pd.testing.assert_frame_equal(result, expected[expected["id"] < 750])


def test_load_extension_left_join(
    small_sky_with_sources_extension_dir, tmp_path, small_sky_with_nested_sources
):
    """Only some of the rows have extension data. The others are kept, with missing values."""
    collection = copy_collection(
        small_sky_with_sources_extension_dir,
        tmp_path,
        update_extension=lambda extension: extension.query("object_id < 750"),
        properties={"extension_join_style": "left"},
    )
    result = lsdb.open_catalog(collection).load_extension("lightcurves").compute()
    assert len(result) == len(small_sky_with_nested_sources) == 131
    expected = small_sky_with_nested_sources.compute()
    has_extension = result["id"] < 750
    pd.testing.assert_frame_equal(result[has_extension], expected[expected["id"] < 750])
    assert result.loc[~has_extension, "sources"].isna().all()


def test_load_two_extensions(small_sky_with_sources_extension_dir, tmp_path, small_sky_with_nested_sources):
    """Two extensions are loaded one after the other."""
    collection = copy_collection(small_sky_with_sources_extension_dir, tmp_path)
    # The second extension has the data of the first, in a column with another name.
    extension = lsdb.open_catalog(collection / "small_sky_order1_lightcurves", columns="all")
    extension.rename({"sources": "detections"}).write_catalog(
        collection / "small_sky_order1_detections", progress_bar=False
    )
    properties = ExtensionProperties.read_from_file(collection / "small_sky_order1_lightcurves.properties")
    properties.model_copy(
        update={
            "name": "small_sky_order1_detections",
            "join_catalog": "small_sky_order1_detections",
            "extension_columns": ["detections"],
        }
    ).to_properties_file(collection)
    collection_properties = CollectionProperties.read_from_dir(collection)
    collection_properties.all_extensions["detections"] = "small_sky_order1_detections.properties"
    collection_properties.to_properties_file(collection)

    catalog = lsdb.open_catalog(collection)
    extended = catalog.load_extension("lightcurves").load_extension("detections")
    assert extended.extensions == ("lightcurves", "detections")
    assert extended.columns.tolist() == [*catalog.columns, "sources", "detections"]
    expected = small_sky_with_nested_sources.compute()
    pd.testing.assert_frame_equal(extended.compute(), expected.assign(detections=expected["sources"]))


def test_load_extension_errors(small_sky_with_sources_extension, small_sky_order1_dir):
    # A path to an extension properties file that does not exist.
    with pytest.raises(FileNotFoundError, match="other_extension.properties"):
        small_sky_with_sources_extension.load_extension("other_extension.properties")
    # A standalone catalog has no extensions.
    with pytest.raises(FileNotFoundError, match="lightcurves"):
        lsdb.open_catalog(small_sky_order1_dir).load_extension("lightcurves")
    with pytest.raises(ValueError, match="is not an extension properties file"):
        small_sky_with_sources_extension.load_extension(small_sky_order1_dir)
    # Loading an extension with the same name twice raises an error.
    catalog = small_sky_with_sources_extension.load_extension("lightcurves")
    with pytest.raises(ValueError, match="is already loaded"):
        catalog.load_extension("lightcurves")


def test_load_extension_storage_options(small_sky_with_sources_extension_dir, mocker):
    """An extension loaded by name uses the storage options of the catalog's collection,
    and one loaded by path uses the storage options given."""
    catalog = lsdb.open_catalog(
        small_sky_with_sources_extension_dir, storage_options={"collection": "option"}
    )
    properties_path = str(small_sky_with_sources_extension_dir / "small_sky_order1_lightcurves.properties")
    read_dataset = mocker.spy(lsdb.loaders.hats.read_hats, "_read_dataset")
    catalog.load_extension("lightcurves")
    catalog.load_extension(properties_path)
    catalog.load_extension(properties_path, storage_options={"given": "option"})
    # The options that the extension data is opened with.
    storage_options = [call.args[0].storage_options for call in read_dataset.call_args_list]
    assert storage_options == [{"collection": "option"}, {}, {"given": "option"}]


def copy_collection(collection_dir, tmp_path, *, update_extension=None, properties=None):
    """Copy the collection, rewriting the extension data with `update_extension(extension)`,
    and updating the extension `properties`."""
    collection = tmp_path / "collection"
    shutil.copytree(collection_dir, collection)
    if update_extension is not None:
        shutil.rmtree(collection / "small_sky_order1_lightcurves")
        extension = lsdb.open_catalog(collection_dir / "small_sky_order1_lightcurves", columns="all")
        update_extension(extension).write_catalog(
            collection / "small_sky_order1_lightcurves", progress_bar=False
        )
    if properties is not None:
        properties_path = collection / "small_sky_order1_lightcurves.properties"
        extension_properties = ExtensionProperties.read_from_file(properties_path)
        extension_properties.model_copy(update=properties).to_properties_file(collection)
    return collection
