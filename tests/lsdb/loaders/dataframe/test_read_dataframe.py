import os
from pathlib import Path

import pandas as pd
import pytest
from hipscat.catalog import CatalogType

import lsdb


def test_read_catalog_from_dataframe(small_sky_order1_catalog, small_sky_order1_csv):
    """Tests that we can initialize a catalog from a Pandas Dataframe
    and that the loaded content is correct"""
    small_sky_order_1_catalog_info = small_sky_order1_catalog.hc_structure.catalog_info

    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.read_dataframe(
        path=small_sky_order1_csv,
        catalog_name=small_sky_order_1_catalog_info.catalog_name,
        catalog_type=small_sky_order_1_catalog_info.catalog_type,
    )
    assert isinstance(catalog, lsdb.Catalog)

    # Catalogs have the same information
    assert catalog.hc_structure.catalog_info == small_sky_order_1_catalog_info

    # Dataframes have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().reset_index(drop=True),
        small_sky_order1_catalog.compute().reset_index(drop=True),
        check_column_type=False,
    )


def test_path_is_valid(tmp_path, small_sky_order1_dir, small_sky_order1_csv):
    """Tests that the path provided to the loader is valid"""
    with pytest.raises(FileNotFoundError, match="Catalog file could not be found"):
        # Path is of a regular file, but it does not exist
        lsdb.read_dataframe(path="catalog.csv", catalog_type=CatalogType.SOURCE)
        # Path exists but it is a directory
        lsdb.read_dataframe(path=small_sky_order1_dir, catalog_type=CatalogType.SOURCE)

    with pytest.raises(ValueError, match="Catalog file is not in CSV format"):
        tmp_txt_file = os.path.join(tmp_path, "catalog.txt")
        Path(tmp_txt_file).touch()
        # Path exists but it does not point to CSV file
        lsdb.read_dataframe(path=tmp_txt_file, catalog_type=CatalogType.SOURCE)

    # Path exists and it is regular file
    catalog = lsdb.read_dataframe(path=small_sky_order1_csv, catalog_type=CatalogType.SOURCE)
    assert isinstance(catalog, lsdb.Catalog)


def test_read_catalog_of_invalid_type(small_sky_order1_csv):
    """Tests that an exception is thrown if the catalog is not of type OBJECT or SOURCE"""
    # Catalog is created for valid types
    valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE]

    for catalog_type in valid_catalog_types:
        lsdb.read_dataframe(path=small_sky_order1_csv, catalog_name="catalog", catalog_type=catalog_type)

    # An error is thrown for all other types
    invalid_catalog_types = [
        catalog_type for catalog_type in CatalogType.all_types() if catalog_type not in valid_catalog_types
    ]
    for catalog_type in invalid_catalog_types:
        with pytest.raises(ValueError, match="Catalog must be of type OBJECT or SOURCE"):
            lsdb.read_dataframe(path=small_sky_order1_csv, catalog_name="catalog", catalog_type=catalog_type)
