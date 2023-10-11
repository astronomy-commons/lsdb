import pandas as pd
import pytest
from hipscat.catalog import CatalogType
from hipscat.pixel_math import HealpixPixel

import lsdb


def get_catalog_kwargs(catalog, **kwargs):
    """Generates arguments for a test catalog. By default, the
    partition size is 1 kB, and it is presented in megabytes."""
    catalog_info = catalog.hc_structure.catalog_info
    kwargs = {
        "catalog_name": catalog_info.catalog_name,
        "catalog_type": catalog_info.catalog_type,
        "partition_size": 1 << 10,
        **kwargs,
    }
    return kwargs


def test_read_catalog_from_dataframe(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that we can initialize a catalog from a Pandas Dataframe and
    that the loaded content is correct"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    assert isinstance(catalog, lsdb.Catalog)
    # Catalogs have the same information
    assert catalog.hc_structure.catalog_info == small_sky_order1_catalog.hc_structure.catalog_info
    # Dataframes have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().reset_index(drop=True),
        small_sky_order1_catalog.compute().reset_index(drop=True),
        check_column_type=False,
    )


def test_read_catalog_of_invalid_type(small_sky_order1_df):
    """Tests that an exception is thrown if the catalog is not of type OBJECT or SOURCE"""
    valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE]
    for catalog_type in CatalogType.all_types():
        kwargs = {"catalog_name": "catalog", "catalog_type": catalog_type}
        if catalog_type in valid_catalog_types:
            lsdb.from_dataframe(small_sky_order1_df, **kwargs)
        else:
            with pytest.raises(ValueError, match="Catalog must be of type OBJECT or SOURCE"):
                lsdb.from_dataframe(small_sky_order1_df, **kwargs)
        # Drop hipscat_index that might have been created in place
        small_sky_order1_df.reset_index(drop=True, inplace=True)


def test_partitions_on_map_equal_partitions_in_df(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions on the partition map exist in the Dask Dataframe"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for _, partition_index in catalog._ddf_pixel_map.items():
        partition_df = catalog._ddf.partitions[partition_index].compute()
        assert isinstance(partition_df, pd.DataFrame)


def test_partitions_in_partition_info_equal_partitions_on_map(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions in the partition info match those on the partition map"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for _, row in catalog.hc_structure.get_pixels().iterrows():
        hp_pixel = HealpixPixel(row["Norder"], row["Npix"])
        partition_from_df = catalog.get_partition(hp_pixel.order, hp_pixel.pixel)
        partition_index = catalog._ddf_pixel_map[hp_pixel]
        partition_from_map = catalog._ddf.partitions[partition_index]
        pd.testing.assert_frame_equal(partition_from_df.compute(), partition_from_map.compute())


def test_partitions_on_map_match_pixel_tree(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that HEALPix pixels on the partition map exist in pixel tree"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for hp_pixel, _ in catalog._ddf_pixel_map.items():
        pixels = catalog.hc_structure.pixel_tree.get_leaf_nodes_at_healpix_pixel(hp_pixel)
        assert len(pixels) > 0


def test_from_dataframe_with_non_default_ra_dec_columns(small_sky_order1_df, small_sky_order1_catalog):
    """Tests the creation of a catalog using non-default ra and dec columns"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, ra_column="my_ra", dec_column="my_dec")
    # If the columns for ra and dec do not exist
    with pytest.raises(KeyError):
        lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # If they were indeed named differently
    small_sky_order1_df.rename(columns={"ra": "my_ra", "dec": "my_dec"}, inplace=True)
    lsdb.from_dataframe(small_sky_order1_df, **kwargs)


def test_partitions_size_is_correct(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified size"""
    partition_size = 1 << 10  # 1 KB in bytes
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(
        small_sky_order1_catalog,
        partition_size=(partition_size / (1 << 20)),  # 1 KB in MB
    )
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # Calculate each partition size
    partition_sizes = [
        partition_df.compute().memory_usage().sum() for partition_df in catalog._ddf.partitions
    ]
    assert all(size <= partition_size for size in partition_sizes)
