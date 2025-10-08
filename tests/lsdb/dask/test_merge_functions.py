import logging

from lsdb.dask.merge_catalog_functions import create_merged_catalog_info


def test_create_merged_catalog_info_suffix_logging(small_sky_catalog, small_sky_xmatch_catalog, caplog):
    with caplog.at_level(logging.WARNING):
        merged_catalog_info = create_merged_catalog_info(
            small_sky_catalog,
            small_sky_xmatch_catalog,
            "merged_catalog",
            ("_left", "_right"),
            suffix_method="overlapping_columns",
        )
        assert merged_catalog_info.catalog_name == "merged_catalog"
        assert merged_catalog_info.ra_column == "ra_left"
        assert merged_catalog_info.dec_column == "dec_left"
        assert "Renaming overlapping columns" not in caplog.text
