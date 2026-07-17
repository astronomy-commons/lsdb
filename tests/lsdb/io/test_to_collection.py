from importlib.metadata import version
from pathlib import Path
from unittest.mock import call

import pandas as pd
import pyarrow.parquet as pq
from hats.io import paths

import lsdb
from lsdb.io.to_hats import write_partitions


def test_save_collection(small_sky_order1_collection_catalog, tmp_path, helpers):
    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"

    small_sky_order1_collection_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
        write_table_kwargs={"compression": "SNAPPY"},
    )

    catalog = lsdb.read_hats(base_collection_path)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1"
    pd.testing.assert_frame_equal(
        catalog.compute(), small_sky_order1_collection_catalog.compute()[["ra", "dec"]]
    )

    helpers.assert_catalog_info_is_correct(
        catalog.hc_structure.catalog_info,
        small_sky_order1_collection_catalog.hc_structure.catalog_info,
        hats_max_rows=42,
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )

    for pixel in catalog.get_healpix_pixels():
        test_path = paths.pixel_catalog_file(catalog.hc_structure.catalog_base_dir, pixel)

        metadata = pq.read_metadata(test_path)
        assert metadata.num_row_groups == 1
        assert metadata.row_group(0).column(0).compression == "SNAPPY"

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_3600arcs"
    pd.testing.assert_frame_equal(
        catalog.margin.compute(), small_sky_order1_collection_catalog.margin.compute()[["ra", "dec"]]
    )
    helpers.assert_catalog_info_is_correct(
        catalog.margin.hc_structure.catalog_info,
        small_sky_order1_collection_catalog.margin.hc_structure.catalog_info,
        catalog_name="small_sky_order1_3600arcs",
        hats_max_rows=7,
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )


def test_save_collection_progress_bar(small_sky_order1_collection_catalog, tmp_path, mocker):
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

    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"

    small_sky_order1_collection_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
        write_table_kwargs={"compression": "SNAPPY"},
    )

    assert tqdm_callback.call_args_list == [
        call(desc="Writing Catalog", disable=False),
        call(desc="Writing Margin Cache", disable=False),
    ]
    assert tqdm_callback.call_count == 2


def test_save_collection_no_progress_bar(small_sky_order1_collection_catalog, tmp_path, mocker):
    tqdm_callback = mocker.patch("lsdb.io.to_hats.TqdmCallback")

    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"

    small_sky_order1_collection_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
        write_table_kwargs={"compression": "SNAPPY"},
        progress_bar=False,
    )

    assert tqdm_callback.call_args_list == [
        call(desc="Writing Catalog", disable=True),
        call(desc="Writing Margin Cache", disable=True),
    ]
    assert tqdm_callback.call_count == 2


def test_save_collection_from_dataframe(small_sky_order1_df, tmp_path):
    expected_catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        margin_threshold=3000,
    )

    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    expected_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
    )

    catalog = lsdb.read_hats(base_collection_path)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1"
    assert catalog.hc_structure.catalog_info.default_columns == ["ra", "dec"]
    assert catalog.hc_structure.catalog_info.obs_regime == "Optical"
    pd.testing.assert_frame_equal(catalog.compute(), expected_catalog.compute()[["ra", "dec"]])

    for pixel in catalog.get_healpix_pixels():
        test_path = paths.pixel_catalog_file(catalog.hc_structure.catalog_base_dir, pixel)

        metadata = pq.read_metadata(test_path)
        assert metadata.num_row_groups == 1
        assert metadata.row_group(0).column(0).compression == "ZSTD"

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_3000arcs"
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 3000
    assert catalog.margin.hc_structure.catalog_info.default_columns == ["ra", "dec"]
    assert catalog.margin.hc_structure.catalog_info.obs_regime == "Optical"
    pd.testing.assert_frame_equal(catalog.margin.compute(), expected_catalog.margin.compute()[["ra", "dec"]])

    for pixel in catalog.margin.get_healpix_pixels():
        test_path = paths.pixel_catalog_file(catalog.margin.hc_structure.catalog_base_dir, pixel)

        metadata = pq.read_metadata(test_path)
        assert metadata.num_row_groups == 1
        assert metadata.row_group(0).column(0).compression == "ZSTD"


def test_save_collection_with_empty_margin(small_sky_order1_df, tmp_path):
    expected_catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        margin_threshold=10,  # a small threshold that produces an empty margin
    )

    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    expected_catalog.write_catalog(base_collection_path, catalog_name="small_sky_order1")

    catalog = lsdb.read_hats(base_collection_path)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1"

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_10arcs"
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 10
    pd.testing.assert_frame_equal(expected_catalog.margin.compute(), catalog.margin.compute())


def test_save_collection_creates_summary_by_default(small_sky_order1_collection_catalog, tmp_path):
    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    small_sky_order1_collection_catalog.write_catalog(base_collection_path, catalog_name="small_sky_order1")
    collection_summary = base_collection_path / "README.md"
    catalog_summary = base_collection_path / "small_sky_order1" / "README.md"
    margin_summary = base_collection_path / "small_sky_order1_3600arcs" / "README.md"
    for summary_path in (collection_summary, catalog_summary, margin_summary):
        assert summary_path.exists()
        assert len(summary_path.read_text()) > 0


def test_save_collection_no_summary(small_sky_order1_collection_catalog, tmp_path):
    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    small_sky_order1_collection_catalog.write_catalog(
        base_collection_path, catalog_name="small_sky_order1", create_summary=False
    )
    assert not (base_collection_path / "README.md").exists()
    assert not (base_collection_path / "small_sky_order1" / "README.md").exists()
    assert not (base_collection_path / "small_sky_order1_3600arcs" / "README.md").exists()


def test_save_collection_provenance(small_sky_order1_df, tmp_path):
    """Test inheriting the provenance of the original catalog, with property like 'hats_creator'."""
    expected_catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        margin_threshold=3000,
    )

    # save the catalog initially with the property and confirm its presence
    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    expected_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical","hats_creator": "LSDB Unit Test"},
    )

    catalog = lsdb.read_hats(base_collection_path)
    assert catalog.hc_structure.catalog_info.obs_regime == "Optical"
    extras = catalog.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" in extras
    assert extras["hats_creator"] == "LSDB Unit Test"

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_info.obs_regime == "Optical"
    extras = catalog.margin.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" in extras
    assert extras["hats_creator"] == "LSDB Unit Test"

    # create a copy of the catalog, without explicitly inheriting the provenance
    round_trip_catalog_path = Path(tmp_path) / "small_sky_drop_provenance"
    catalog.write_catalog(
        round_trip_catalog_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
    )

    round_trip_catalog = lsdb.read_hats(round_trip_catalog_path)
    assert round_trip_catalog.hc_structure.catalog_info.obs_regime == "Optical"
    extras = round_trip_catalog.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" not in extras

    assert round_trip_catalog.margin is not None
    assert round_trip_catalog.margin.hc_structure.catalog_info.obs_regime == "Optical"
    extras = round_trip_catalog.margin.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" not in extras

    # create a copy of the catalog, *explicitly* inheriting the provenance
    round_trip_catalog_path = Path(tmp_path) / "small_sky_inherit_provenance"
    catalog.write_catalog(
        round_trip_catalog_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        inherit_provenance=True,
    )

    round_trip_catalog = lsdb.read_hats(round_trip_catalog_path)
    assert round_trip_catalog.hc_structure.catalog_info.obs_regime == "Optical"
    extras = round_trip_catalog.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" in extras
    assert extras["hats_creator"] == "LSDB Unit Test"

    assert round_trip_catalog.margin is not None
    assert round_trip_catalog.margin.hc_structure.catalog_info.obs_regime == "Optical"
    extras = round_trip_catalog.margin.hc_structure.catalog_info.extra_dict()
    assert "hats_creator" in extras
    assert extras["hats_creator"] == "LSDB Unit Test"
