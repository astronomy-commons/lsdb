from importlib.metadata import version
from pathlib import Path

import hats as hc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hats.io.file_io import get_upath_for_protocol, read_fits_image
from hats.io.paths import get_data_thumbnail_pointer

import lsdb


def test_save_catalog(small_sky_catalog, tmp_path, helpers):
    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.to_hats(
        base_catalog_path, catalog_name=new_catalog_name, addl_hats_properties={"obs_regime": "Optical"}
    )

    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.schema.pandas_metadata is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_catalog._ddf.compute())

    # When saving a catalog with to_hats, we update the hats_max_rows
    # to the maximum count of points per partition. In this case there
    # is only one with 131 rows, so that is the value we expect.
    partition_sizes = small_sky_catalog._ddf.map_partitions(len).compute()
    assert max(partition_sizes) == 131

    helpers.assert_catalog_info_is_correct(
        expected_catalog.hc_structure.catalog_info,
        small_sky_catalog.hc_structure.catalog_info,
        hats_max_rows="131",
        skymap_order=5,
        obs_regime="Optical",
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )

    # The catalog has 1 partition, therefore the thumbnail has 1 row
    main_catalog_path = base_catalog_path / new_catalog_name
    data_thumbnail_pointer = get_data_thumbnail_pointer(main_catalog_path)
    assert data_thumbnail_pointer.exists()
    data_thumbnail = pq.read_table(data_thumbnail_pointer)
    assert len(data_thumbnail) == 1
    assert data_thumbnail.schema.equals(small_sky_catalog.hc_structure.schema)
    assert (main_catalog_path / "properties").exists()
    assert (main_catalog_path / "hats.properties").exists()


def test_save_catalog_initializes_upath_once(small_sky_catalog, tmp_path, mocker):
    mock_method = "hats.io.file_io.file_pointer.get_upath_for_protocol"
    mocked_upath_call = mocker.patch(mock_method, side_effect=get_upath_for_protocol)

    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.to_hats(base_catalog_path, catalog_name=new_catalog_name)

    mocked_upath_call.assert_called_once_with(base_catalog_path)


def test_save_catalog_default_columns(small_sky_order1_default_cols_catalog, tmp_path, helpers):
    default_columns = ["ra", "dec"]
    cat = small_sky_order1_default_cols_catalog[default_columns]
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    cat.to_hats(base_catalog_path, catalog_name=new_catalog_name, default_columns=default_columns)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    assert expected_catalog.hc_structure.catalog_info.default_columns == default_columns
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat.compute())
    helpers.assert_schema_correct(expected_catalog)
    helpers.assert_default_columns_in_columns(expected_catalog)


def test_save_catalog_empty_default_columns(small_sky_order1_default_cols_catalog, tmp_path, helpers):
    cat = small_sky_order1_default_cols_catalog[["ra", "dec"]]
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    cat.to_hats(base_catalog_path, catalog_name=new_catalog_name, default_columns=[])
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_info.default_columns is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat._ddf.compute())
    helpers.assert_schema_correct(expected_catalog)
    helpers.assert_default_columns_in_columns(expected_catalog)


def test_save_catalog_invalid_default_columns(small_sky_order1_default_cols_catalog, tmp_path):
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    with pytest.raises(ValueError, match="not found"):
        small_sky_order1_default_cols_catalog.to_hats(
            base_catalog_path, catalog_name=new_catalog_name, default_columns=["id", "abc"]
        )


def test_save_crossmatch_catalog(
    small_sky_order1_default_cols_catalog, small_sky_xmatch_catalog, tmp_path, helpers
):
    cat = small_sky_order1_default_cols_catalog.crossmatch(
        small_sky_xmatch_catalog, radius_arcsec=0.01 * 3600
    )
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    # Before the catalog is serialized, verify that it won't have original schema.
    # This is something that will be available after serialization.
    with pytest.raises(ValueError, match="Original catalog schema is not available"):
        _ = cat.original_schema
    cat.to_hats(base_catalog_path, catalog_name=new_catalog_name)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.original_schema is not None
    assert expected_catalog.hc_structure.catalog_info.default_columns is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat.compute())
    helpers.assert_schema_correct(expected_catalog)


def test_save_catalog_point_map(small_sky_order1_catalog, tmp_path):
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name

    small_sky_order1_catalog.to_hats(
        base_catalog_path,
        catalog_name=new_catalog_name,
        skymap_alt_orders=[1, 2],
        histogram_order=8,
    )

    main_catalog_path = base_catalog_path / new_catalog_name

    point_map_path = main_catalog_path / "point_map.fits"
    assert point_map_path.exists()
    histogram = read_fits_image(point_map_path)

    # The histogram and the sky map histogram match
    assert len(small_sky_order1_catalog) == np.sum(histogram)

    skymap_path = main_catalog_path / "skymap.fits"
    assert skymap_path.exists()
    skymap_histogram = read_fits_image(skymap_path)

    # The histogram and the sky map histogram match
    assert len(small_sky_order1_catalog) == np.sum(skymap_histogram)
    npt.assert_array_equal(histogram, skymap_histogram)

    skymap_path = main_catalog_path / "skymap.1.fits"
    assert skymap_path.exists()

    skymap_path = main_catalog_path / "skymap.2.fits"
    assert skymap_path.exists()

    new_catalog = lsdb.open_catalog(base_catalog_path)
    assert new_catalog.hc_structure.catalog_info.skymap_alt_orders == [1, 2]
    assert new_catalog.hc_structure.catalog_info.skymap_order == 8


def test_save_catalog_overwrite(small_sky_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    # Saving a catalog to disk when the directory does not yet exist
    small_sky_catalog.to_hats(base_catalog_path)
    # The output directory exists and it has content. Overwrite is
    # set to False and, as such, the operation fails.
    with pytest.raises(ValueError, match="set overwrite to True"):
        small_sky_catalog.to_hats(base_catalog_path)
    # With overwrite it succeeds because the directory is recreated
    small_sky_catalog.to_hats(base_catalog_path, overwrite=True)


def test_save_catalog_when_catalog_is_empty(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"

    # The result of this cone search is known to be empty
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 1)
    assert cone_search_catalog._ddf.npartitions == 1

    non_empty_pixels = []
    for pixel, partition_index in cone_search_catalog._ddf_pixel_map.items():
        if len(cone_search_catalog._ddf.partitions[partition_index]) > 0:
            non_empty_pixels.append(pixel)
    assert len(non_empty_pixels) == 0

    # The catalog is not written to disk
    with pytest.raises(RuntimeError, match="The output catalog is empty"):
        cone_search_catalog.to_hats(base_catalog_path)


def test_save_big_catalog(tmp_path):
    """Load a catalog with many partitions, and save with to_hats."""
    mock_partition_df = pd.DataFrame(
        {
            "ra": np.linspace(0, 360, 100_000),
            "dec": np.linspace(-90, 90, 100_000),
            "id": np.arange(100_000, 200_000),
        }
    )

    base_catalog_path = tmp_path / "big_sky"

    kwargs = {
        "catalog_name": "big_sky",
        "catalog_type": "object",
        "lowest_order": 6,
        "highest_order": 10,
        "threshold": 500,
    }

    catalog = lsdb.from_dataframe(mock_partition_df, margin_threshold=None, **kwargs)

    catalog.to_hats(base_catalog_path)

    read_catalog = hc.read_hats(base_catalog_path)
    assert len(read_catalog.get_healpix_pixels()) == len(catalog.get_healpix_pixels())


def test_save_catalog_with_some_empty_partitions(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"

    # The result of this cone search is known to have one empty partition
    cone_search_catalog = small_sky_order1_catalog.cone_search(0, -80, 15 * 3600)
    assert cone_search_catalog._ddf.npartitions == 2

    non_empty_pixels = []
    for pixel, partition_index in cone_search_catalog._ddf_pixel_map.items():
        if len(cone_search_catalog._ddf.partitions[partition_index]) > 0:
            non_empty_pixels.append(pixel)
    assert len(non_empty_pixels) == 1

    cone_search_catalog.to_hats(base_catalog_path)

    # Confirm that we can read the catalog from disk, and that it was
    # written with no empty partitions
    catalog = lsdb.read_hats(base_catalog_path)
    assert catalog._ddf.npartitions == 1
    assert len(catalog._ddf.partitions[0]) > 0
    assert list(catalog._ddf_pixel_map.keys()) == non_empty_pixels
