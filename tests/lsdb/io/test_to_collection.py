from importlib.metadata import version
from pathlib import Path

import lsdb


def test_save_collection(small_sky_order1_collection_catalog, tmp_path, helpers):
    collection_name = "small_sky_order1_collection"
    base_collection_path = Path(tmp_path) / collection_name

    small_sky_order1_collection_catalog.to_collection(
        base_collection_path,
        collection_name=collection_name,
        default_columns=["ra", "dec"],
        addl_hats_properties={"obs_regime": "Optical"},
    )

    catalog = lsdb.read_hats(base_collection_path)
    assert isinstance(catalog, lsdb.Catalog)
    assert catalog.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1"
    helpers.assert_catalog_info_is_correct(
        catalog.hc_structure.catalog_info,
        small_sky_order1_collection_catalog.hc_structure.catalog_info,
        hats_max_rows="42",
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )

    assert catalog.margin is not None
    assert (
        catalog.margin.hc_structure.catalog_base_dir == base_collection_path / "small_sky_order1_margin_1deg"
    )
    helpers.assert_catalog_info_is_correct(
        catalog.margin.hc_structure.catalog_info,
        small_sky_order1_collection_catalog.margin.hc_structure.catalog_info,
        hats_max_rows="7",
        obs_regime="Optical",
        default_columns=["ra", "dec"],
        hats_builder=f"lsdb v{version('lsdb')}, hats v{version('hats')}",
    )
