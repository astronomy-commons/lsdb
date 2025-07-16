from importlib.metadata import version
from pathlib import Path

import pandas as pd

import lsdb


def test_save_collection(small_sky_order1_collection_catalog, tmp_path, helpers):
    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"

    small_sky_order1_collection_catalog.write_catalog(
        base_collection_path,
        catalog_name="small_sky_order1",
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
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
        hats_max_rows="42",
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_3600arcs"
    pd.testing.assert_frame_equal(
        catalog.margin.compute(), small_sky_order1_collection_catalog.margin.compute()[["ra", "dec"]]
    )
    helpers.assert_catalog_info_is_correct(
        catalog.margin.hc_structure.catalog_info,
        small_sky_order1_collection_catalog.margin.hc_structure.catalog_info,
        catalog_name="small_sky_order1_3600arcs",
        hats_max_rows="7",
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )


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

    assert catalog.margin is not None
    assert catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_3000arcs"
    assert catalog.margin.hc_structure.catalog_info.margin_threshold == 3000
    assert catalog.margin.hc_structure.catalog_info.default_columns == ["ra", "dec"]
    assert catalog.margin.hc_structure.catalog_info.obs_regime == "Optical"
    pd.testing.assert_frame_equal(catalog.margin.compute(), expected_catalog.margin.compute()[["ra", "dec"]])


def test_save_collection_with_empty_margin(small_sky_order1_df, tmp_path):
    catalog = lsdb.from_dataframe(
        small_sky_order1_df,
        catalog_name="small_sky_order1",
        catalog_type="object",
        margin_threshold=10,  # a small threshold that produces an empty margin
    )

    base_collection_path = Path(tmp_path) / "small_sky_order1_collection"
    catalog.write_catalog(base_collection_path, catalog_name="small_sky_order1")

    catalog = lsdb.read_hats(base_collection_path)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1"
    assert catalog.margin is None
