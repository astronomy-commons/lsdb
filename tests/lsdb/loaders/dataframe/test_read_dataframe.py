import pandas as pd
import pytest
from hipscat.catalog import CatalogType

import lsdb


def test_read_catalog_from_dataframe(small_sky_order1_catalog, small_sky_order1_csv):
    """Tests that we can initialize a catalog from a Pandas Dataframe
    and that the loaded content is correct."""
    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.read_dataframe(
        path=small_sky_order1_csv,
        catalog_name="small_sky_order1",
        catalog_type=CatalogType.SOURCE
    )
    assert isinstance(catalog, lsdb.Catalog)

    # Catalogs have the same information
    assert catalog.hc_structure.catalog_info == small_sky_order1_catalog.hc_structure.catalog_info

    # Data frames have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().reset_index(drop=True),
        small_sky_order1_catalog.compute().reset_index(drop=True),
        check_column_type=False
    )


def test_read_catalog_dataframe_raises_file_not_found(small_sky_order1_dir, small_sky_order1_csv):
    """Tests that the path provided to the loader is valid"""
    with pytest.raises(FileNotFoundError):
        # Path is of a regular file, but it does not exist
        lsdb.read_dataframe("mock.csv")
        # Path exists but it is a directory
        lsdb.read_dataframe(small_sky_order1_dir)

    # Path exists and it is regular file
    catalog = lsdb.read_dataframe(small_sky_order1_csv)
    assert isinstance(catalog, lsdb.Catalog)
