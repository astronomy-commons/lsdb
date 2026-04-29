import shutil
from importlib.metadata import version
from pathlib import Path

import dask
import hats as hc
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest
from dask.diagnostics import Profiler
from hats.catalog import PartitionInfo, TableProperties
from hats.io.file_io import get_upath_for_protocol, read_fits_image
from hats.io.paths import get_data_thumbnail_pointer
from hats.pixel_math.sparse_histogram import SparseHistogram
from hats.testing import assert_catalog_info_is_correct
from pydantic import ValidationError

import lsdb
from lsdb.io.common import set_default_write_table_kwargs
from lsdb.io.to_hats import (
    DONE_DIR_NAME,
    HISTOGRAM_DIR_NAME,
    perform_write,
    write_histogram,
    write_partitions,
)


def test_save_catalog(small_sky_catalog, tmp_path, helpers):
    new_catalog_name = "small_sky"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    small_sky_catalog.write_catalog(
        base_catalog_path, catalog_name=new_catalog_name, addl_hats_properties={"obs_regime": "Optical"}
    )

    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.schema.pandas_metadata is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == small_sky_catalog.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), small_sky_catalog._ddf.compute())

    # When saving a catalog with write_catalog, we update the hats_max_rows
    # to the maximum count of points per partition. In this case there
    # is only one with 131 rows, so that is the value we expect.
    partition_sizes = small_sky_catalog._ddf.map_partitions(len).compute()
    assert max(partition_sizes) == 131

    helpers.assert_catalog_info_is_correct(
        expected_catalog.hc_structure.catalog_info,
        small_sky_catalog.hc_structure.catalog_info,
        hats_max_rows=131,
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
    small_sky_catalog.write_catalog(base_catalog_path, catalog_name=new_catalog_name)

    mocked_upath_call.assert_called_once_with(base_catalog_path)


def test_save_catalog_default_columns(small_sky_with_nested_sources, tmp_path, helpers):
    # Including the entirety of "sources" which is a nested column
    default_columns = ["ra", "dec", "sources"]
    cat = small_sky_with_nested_sources[default_columns]
    new_catalog_name = "small_sky_order1_nested_sources"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    cat.write_catalog(base_catalog_path, catalog_name=new_catalog_name, default_columns=default_columns)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    assert expected_catalog.hc_structure.catalog_info.default_columns == default_columns
    assert len(expected_catalog["sources"].nest.columns) == 8
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat.compute())
    helpers.assert_schema_correct(expected_catalog)
    helpers.assert_default_columns_in_columns(expected_catalog)


def test_save_catalog_default_nested_columns(small_sky_with_nested_sources, tmp_path, helpers):
    # Selecting just some of the nested columns
    default_columns = ["ra", "dec", "sources.mjd", "sources.mag"]
    cat = small_sky_with_nested_sources[default_columns]
    new_catalog_name = "small_sky_order1_nested_sources"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    cat.write_catalog(base_catalog_path, catalog_name=new_catalog_name, default_columns=default_columns)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    assert expected_catalog.hc_structure.catalog_info.default_columns == default_columns
    assert len(expected_catalog["sources"].nest.columns) == 2
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat.compute())
    helpers.assert_schema_correct(expected_catalog)
    helpers.assert_default_columns_in_columns(expected_catalog)


def test_save_catalog_shows_progress_bar(small_sky_catalog, tmp_path, mocker):
    in_progress = {"active": False}

    class _ProgressContext:
        def __enter__(self):
            in_progress["active"] = True

        def __exit__(self, exc_type, exc, tb):
            in_progress["active"] = False

    tqdm_callback = mocker.patch(
        "lsdb.io.to_hats.TqdmCallback",
        return_value=_ProgressContext(),
    )

    def _checked_write_partitions(*args, **kwargs):
        assert in_progress["active"]
        return write_partitions(*args, **kwargs)

    mocker.patch("lsdb.io.to_hats.write_partitions", side_effect=_checked_write_partitions)
    base_catalog_path = Path(tmp_path) / "small_sky"

    small_sky_catalog.write_catalog(base_catalog_path, progress_bar=True)

    tqdm_callback.assert_called_once_with(desc="Writing Catalog", disable=False)


def test_save_catalog_without_progress_bar(small_sky_catalog, tmp_path, mocker):
    tqdm_callback = mocker.patch("lsdb.io.to_hats.TqdmCallback")
    base_catalog_path = Path(tmp_path) / "small_sky"

    small_sky_catalog.write_catalog(base_catalog_path, progress_bar=False)

    tqdm_callback.assert_called_once_with(desc="Writing Catalog", disable=True)


def test_save_catalog_progress_bar_kwargs(small_sky_catalog, tmp_path, mocker):
    tqdm_callback = mocker.patch("lsdb.io.to_hats.TqdmCallback")
    base_catalog_path = Path(tmp_path) / "small_sky"

    small_sky_catalog.write_catalog(
        base_catalog_path, tqdm_kwargs={"desc": "Custom Description", "ascii": True}
    )

    tqdm_callback.assert_called_once_with(desc="Custom Description", ascii=True, disable=False)


def test_save_catalog_empty_default_columns(small_sky_order1_default_cols_catalog, tmp_path, helpers):
    cat = small_sky_order1_default_cols_catalog[["ra", "dec"]]
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    cat.write_catalog(base_catalog_path, catalog_name=new_catalog_name, default_columns=[])
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.hc_structure.catalog_info.default_columns is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat._ddf.compute())
    helpers.assert_schema_correct(expected_catalog)
    helpers.assert_default_columns_in_columns(expected_catalog)


def test_save_catalog_with_npix_suffix(small_sky_order1_collection_catalog, tmp_path):
    small_sky_order1_collection_catalog.write_catalog(
        tmp_path / "small_sky_collection",
        npix_suffix="/",
        npix_parquet_name="data.pq",
    )

    collection_path = tmp_path / "small_sky_collection"
    catalog = lsdb.read_hats(collection_path)

    pixel_path = "dataset/Norder=1/Dir=0/Npix=44/data.pq"
    assert catalog.hc_structure.catalog_info.npix_suffix == "/"
    assert (collection_path / "small_sky_order1" / pixel_path).exists()
    assert catalog.margin.hc_structure.catalog_info.npix_suffix == "/"
    assert (collection_path / "small_sky_order1_3600arcs" / pixel_path).exists()

    expected_cat = small_sky_order1_collection_catalog.compute()
    pd.testing.assert_frame_equal(expected_cat, catalog.compute())
    expected_margin = small_sky_order1_collection_catalog.margin.compute()
    pd.testing.assert_frame_equal(expected_margin, catalog.margin.compute())


def test_save_catalog_invalid_default_columns(small_sky_order1_default_cols_catalog, tmp_path):
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    with pytest.raises(ValueError, match="not found"):
        small_sky_order1_default_cols_catalog.write_catalog(
            base_catalog_path, catalog_name=new_catalog_name, default_columns=["id", "abc"]
        )


def test_save_catalog_invalid_default_nested_columns(small_sky_with_nested_sources, tmp_path):
    new_catalog_name = "small_sky_order1_nested_sources"
    base_catalog_path = Path(tmp_path) / new_catalog_name
    # Cannot specify partial and full load of a column
    with pytest.raises(ValueError, match="'sources'"):
        small_sky_with_nested_sources.write_catalog(
            base_catalog_path,
            catalog_name=new_catalog_name,
            default_columns=["ra", "dec", "sources", "sources.mjd"],
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
    cat.write_catalog(base_catalog_path, catalog_name=new_catalog_name)
    expected_catalog = lsdb.read_hats(base_catalog_path)
    assert expected_catalog.original_schema is not None
    assert expected_catalog.hc_structure.catalog_info.default_columns is None
    assert expected_catalog.hc_structure.catalog_name == new_catalog_name
    assert expected_catalog.get_healpix_pixels() == cat.get_healpix_pixels()
    pd.testing.assert_frame_equal(expected_catalog.compute(), cat.compute())
    helpers.assert_schema_correct(expected_catalog)


def test_save_catalog_point_map(small_sky_order1_df, tmp_path):
    new_catalog_name = "small_sky_order1"
    base_catalog_path = Path(tmp_path) / new_catalog_name

    small_sky_order1_catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        margin_threshold=None,
        catalog_name="small_sky_order1",
        lowest_order=6,
        highest_order=8,
        partition_rows=500,
    )

    small_sky_order1_catalog.write_catalog(
        base_catalog_path,
        catalog_name=new_catalog_name,
        skymap_alt_orders=[1, 2],
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
    small_sky_catalog.write_catalog(base_catalog_path)
    # The output directory exists and it has content. Overwrite is
    # set to False and, as such, the operation fails.
    with pytest.raises(ValueError, match="set overwrite"):
        small_sky_catalog.write_catalog(base_catalog_path)
    # With overwrite it succeeds because the directory is recreated
    small_sky_catalog.write_catalog(base_catalog_path, overwrite=True)


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
        cone_search_catalog.write_catalog(base_catalog_path)


def test_save_empty_catalog_no_error(small_sky_order1_catalog, tmp_path):
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
    cone_search_catalog.write_catalog(base_catalog_path, error_if_empty=False)

    catalog = lsdb.read_hats(base_catalog_path)
    assert len(catalog.get_healpix_pixels()) == 0
    assert len(catalog.per_partition_statistics()) == 0
    pd.testing.assert_frame_equal(cone_search_catalog._ddf._meta, catalog._ddf._meta)
    pd.testing.assert_frame_equal(cone_search_catalog.compute(), catalog.compute())


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

    cone_search_catalog.write_catalog(base_catalog_path)

    # Confirm that we can read the catalog from disk, and that it was
    # written with no empty partitions
    catalog = lsdb.read_hats(base_catalog_path)
    assert catalog._ddf.npartitions == 1
    assert len(catalog._ddf.partitions[0]) > 0
    assert list(catalog._ddf_pixel_map.keys()) == non_empty_pixels


def _write_partial_catalog(catalog, path, pixels_to_write):
    partitions = catalog._ddf.to_delayed()
    results, pixels = [], []
    for pixel in pixels_to_write:
        partition_index = catalog._ddf_pixel_map[pixel]
        results.append(
            perform_write(
                partitions[partition_index],
                pixel,
                path,
                catalog.hc_structure.catalog_info.skymap_order,
                **set_default_write_table_kwargs(None),
            )
        )
        pixels.append(pixel)
    dask.compute(*results)


def _copy_metadata_files(reference_catalog_path, base_catalog_path):
    for entry in reference_catalog_path.iterdir():
        if entry.name == "dataset":
            continue
        target = base_catalog_path / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(entry, target)


def _assert_catalog_matches_reference(catalog, ref_cat):
    assert catalog.get_healpix_pixels() == ref_cat.get_healpix_pixels()
    assert_catalog_info_is_correct(ref_cat.hc_structure.catalog_info, catalog.hc_structure.catalog_info)
    assert catalog.hc_structure.moc == ref_cat.hc_structure.moc
    pd.testing.assert_frame_equal(catalog.compute(), ref_cat.compute())
    pd.testing.assert_frame_equal(catalog.per_partition_statistics(), ref_cat.per_partition_statistics())


def test_resume_catalog_write(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    # Write only some of the partitions to disk
    pixels_to_write = small_sky_order1_catalog.get_healpix_pixels()[:2]
    _write_partial_catalog(small_sky_order1_catalog, base_catalog_path, pixels_to_write)

    # Resume the write and confirm that all partitions are written to disk
    with Profiler() as prof:
        small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)
    total_resume_tasks_num = len(prof.results)  # pylint: disable=no-member
    with Profiler() as prof:
        small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)
    total_ref_tasks_num = len(prof.results)  # pylint: disable=no-member

    assert total_resume_tasks_num < total_ref_tasks_num

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)

    done_dir = base_catalog_path / DONE_DIR_NAME
    assert not done_dir.exists()

    hists_dir = base_catalog_path / HISTOGRAM_DIR_NAME
    assert not hists_dir.exists()

    assert [p.relative_to(base_catalog_path) for p in base_catalog_path.rglob("*")] == [
        p.relative_to(reference_catalog_path) for p in reference_catalog_path.rglob("*")
    ]


def test_resume_catalog_collection_write(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    # Write only some of the partitions to disk
    pixels_to_write = small_sky_order1_catalog.get_healpix_pixels()[:2]
    collection_catalog_path = base_catalog_path / small_sky_order1_catalog.hc_structure.catalog_name
    _write_partial_catalog(small_sky_order1_catalog, collection_catalog_path, pixels_to_write)

    # Resume the write and confirm that all partitions are written to disk
    with Profiler() as prof:
        small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True)
    total_resume_tasks_num = len(prof.results)  # pylint: disable=no-member
    with Profiler() as prof:
        small_sky_order1_catalog.write_catalog(reference_catalog_path)
    total_ref_tasks_num = len(prof.results)  # pylint: disable=no-member

    assert total_resume_tasks_num < total_ref_tasks_num

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_errors_with_different_catalog(small_sky_order1_catalog, small_sky_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    _write_partial_catalog(small_sky_catalog, base_catalog_path, small_sky_catalog.get_healpix_pixels())

    with pytest.raises(ValueError, match="not present in the provided catalog"):
        small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)


def test_resume_errors_with_different_hists_and_done(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    _write_partial_catalog(
        small_sky_order1_catalog, base_catalog_path, small_sky_order1_catalog.get_healpix_pixels()[:2]
    )
    hist_pixel = small_sky_order1_catalog.get_healpix_pixels()[2]
    write_histogram(SparseHistogram([], [], 8), base_catalog_path, hist_pixel)

    with pytest.raises(ValueError, match="histogram files .* do not match the done pixels"):
        small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)


def test_resume_errors_with_invalid_parquet(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    _write_partial_catalog(
        small_sky_order1_catalog, base_catalog_path, small_sky_order1_catalog.get_healpix_pixels()[:2]
    )
    next_pixel = small_sky_order1_catalog.get_healpix_pixels()[2]
    parquet_path = hc.io.paths.pixel_catalog_file(base_catalog_path, next_pixel)
    hc.io.file_io.make_directory(parquet_path.parent, exist_ok=True)
    with parquet_path.open("w") as f:
        f.write("This is not a valid parquet file")

    small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)
    small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_catalog_write_all_files_written(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    _write_partial_catalog(
        small_sky_order1_catalog, base_catalog_path, small_sky_order1_catalog.get_healpix_pixels()
    )

    small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)
    small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_catalog_write_with_parquet_metadata_files_written(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    pixels_to_write = small_sky_order1_catalog.get_healpix_pixels()
    _write_partial_catalog(small_sky_order1_catalog, base_catalog_path, pixels_to_write)
    hc.io.write_parquet_metadata(
        base_catalog_path,
    )

    small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)
    small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_catalog_write_with_all_metadata_files_written(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)

    pixels_to_write = small_sky_order1_catalog.get_healpix_pixels()
    _write_partial_catalog(small_sky_order1_catalog, base_catalog_path, pixels_to_write)
    _copy_metadata_files(reference_catalog_path, base_catalog_path)

    small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_catalog_write_with_corrupted_metadata_files_written(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"
    reference_catalog_path = tmp_path / "reference_small_sky"

    small_sky_order1_catalog.write_catalog(reference_catalog_path, as_collection=False)

    pixels_to_write = small_sky_order1_catalog.get_healpix_pixels()
    _write_partial_catalog(small_sky_order1_catalog, base_catalog_path, pixels_to_write)

    properties_path = base_catalog_path / "hats.properties"
    with properties_path.open("w") as f:
        f.write("This is not a valid properties file")

    partition_info_path = base_catalog_path / "partition_info.csv"
    with partition_info_path.open("w") as f:
        f.write("This is not a valid partition info file")

    with pytest.raises(ValidationError):
        TableProperties.read_from_dir(base_catalog_path)

    with pytest.raises(KeyError):
        PartitionInfo.read_from_dir(base_catalog_path)

    small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, as_collection=False)

    catalog = lsdb.read_hats(base_catalog_path)
    ref_cat = lsdb.read_hats(reference_catalog_path)
    _assert_catalog_matches_reference(catalog, ref_cat)


def test_resume_and_overwrite_catalog_write(small_sky_order1_catalog, tmp_path):
    base_catalog_path = tmp_path / "small_sky"

    with pytest.raises(ValueError, match="overwrite and resume cannot both be True"):
        small_sky_order1_catalog.write_catalog(base_catalog_path, resume=True, overwrite=True)
