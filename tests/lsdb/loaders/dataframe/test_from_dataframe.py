import healpy as hp
import pandas as pd
import pytest
from conftest import assert_divisions_are_correct
from hipscat.catalog import CatalogType
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_tree.pixel_node_type import PixelNodeType

import lsdb
from lsdb.loaders.dataframe.dataframe_catalog_loader import DataframeCatalogLoader


def get_catalog_kwargs(catalog, **kwargs):
    """Generates arguments for a test catalog. By default, the
    partition size is 1 kB, and it is presented in megabytes."""
    catalog_info = catalog.hc_structure.catalog_info
    kwargs = {
        "catalog_name": catalog_info.catalog_name,
        "catalog_type": catalog_info.catalog_type,
        "highest_order": 5,
        "threshold": 50,
        **kwargs,
    }
    return kwargs


def test_from_dataframe(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that we can initialize a catalog from a Pandas Dataframe and
    that the loaded content is correct"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    # Read CSV file for the small sky order 1 catalog
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    assert isinstance(catalog, lsdb.Catalog)
    # Catalogs have the same information
    assert catalog.hc_structure.catalog_info == small_sky_order1_catalog.hc_structure.catalog_info
    # Index is set to hipscat index
    assert catalog._ddf.index.name == HIPSCAT_ID_COLUMN
    # Dataframes have the same data (column data types may differ)
    pd.testing.assert_frame_equal(
        catalog.compute().sort_index(),
        small_sky_order1_catalog.compute().sort_index(),
        check_dtype=False,
    )
    # Divisions belong to the respective HEALPix pixels
    assert_divisions_are_correct(catalog)


def test_from_dataframe_catalog_of_invalid_type(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that an exception is thrown if the catalog is not of type OBJECT or SOURCE"""
    valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE]
    for catalog_type in CatalogType.all_types():
        kwargs = get_catalog_kwargs(small_sky_order1_catalog, catalog_type=catalog_type)
        if catalog_type in valid_catalog_types:
            lsdb.from_dataframe(small_sky_order1_df, **kwargs)
        else:
            with pytest.raises(ValueError, match="Catalog must be of type OBJECT or SOURCE"):
                lsdb.from_dataframe(small_sky_order1_df, **kwargs)
        # Drop hipscat_index that might have been created in place
        small_sky_order1_df.reset_index(drop=True, inplace=True)


def test_from_dataframe_when_threshold_and_partition_size_specified(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that specifying simultaneously threshold and partition_size is invalid"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=10, threshold=10_000)
    with pytest.raises(ValueError, match="Specify only one: threshold or partition_size"):
        lsdb.from_dataframe(small_sky_order1_df, **kwargs)


def test_partitions_on_map_equal_partitions_in_df(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions on the partition map exist in the Dask Dataframe"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for hp_pixel, partition_index in catalog._ddf_pixel_map.items():
        partition_df = catalog._ddf.partitions[partition_index].compute()
        assert isinstance(partition_df, pd.DataFrame)
        for _, row in partition_df.iterrows():
            ipix = hp.ang2pix(2**hp_pixel.order, row["ra"], row["dec"], nest=True, lonlat=True)
            assert ipix == hp_pixel.pixel


def test_partitions_in_partition_info_equal_partitions_on_map(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions in the partition info match those on the partition map"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for hp_pixel in catalog.hc_structure.get_healpix_pixels():
        partition_from_df = catalog.get_partition(hp_pixel.order, hp_pixel.pixel)
        partition_index = catalog._ddf_pixel_map[hp_pixel]
        partition_from_map = catalog._ddf.partitions[partition_index]
        pd.testing.assert_frame_equal(partition_from_df.compute(), partition_from_map.compute())


def test_partitions_on_map_match_pixel_tree(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that HEALPix pixels on the partition map exist in pixel tree"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    for hp_pixel, _ in catalog._ddf_pixel_map.items():
        if hp_pixel in catalog.hc_structure.pixel_tree:
            assert catalog.hc_structure.pixel_tree[hp_pixel].node_type == PixelNodeType.LEAF


def test_from_dataframe_with_non_default_ra_dec_columns(small_sky_order1_df, small_sky_order1_catalog):
    """Tests the creation of a catalog using non-default ra and dec columns"""
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, ra_column="my_ra", dec_column="my_dec")
    # If the columns for ra and dec do not exist
    with pytest.raises(KeyError):
        lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # If they were indeed named differently
    small_sky_order1_df.rename(columns={"ra": "my_ra", "dec": "my_dec"}, inplace=True)
    lsdb.from_dataframe(small_sky_order1_df, **kwargs)


def test_partitions_obey_partition_size(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified size"""
    # Use partitions with 10 rows
    partition_size = 10
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=partition_size, threshold=None)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # Calculate size of dataframe per partition
    partition_sizes = [len(partition_df) for partition_df in catalog._ddf.partitions]
    assert all(size <= partition_size for size in partition_sizes)


def test_partitions_obey_threshold(small_sky_order1_df, small_sky_order1_catalog):
    """Tests that partitions are limited by the specified threshold"""
    threshold = 50
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, partition_size=None, threshold=threshold)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # Calculate number of pixels per partition
    num_partition_pixels = [len(partition_df.compute().index) for partition_df in catalog._ddf.partitions]
    assert all(num_pixels <= threshold for num_pixels in num_partition_pixels)


def test_partitions_obey_default_threshold_when_no_arguments_specified(
    small_sky_order1_df, small_sky_order1_catalog
):
    """Tests that partitions are limited by the default threshold
    when no partition size or threshold is specified"""
    default_threshold = DataframeCatalogLoader.DEFAULT_THRESHOLD
    # Read CSV file for the small sky order 1 catalog
    kwargs = get_catalog_kwargs(small_sky_order1_catalog, threshold=None, partition_size=None)
    catalog = lsdb.from_dataframe(small_sky_order1_df, **kwargs)
    # Calculate number of pixels per partition
    num_partition_pixels = [len(partition_df.compute().index) for partition_df in catalog._ddf.partitions]
    assert all(num_pixels <= default_threshold for num_pixels in num_partition_pixels)
