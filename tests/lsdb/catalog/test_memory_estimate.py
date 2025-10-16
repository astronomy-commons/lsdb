"""Tests for memory estimation and partition metadata methods."""

import pytest
import pandas as pd
from hats.pixel_math import HealpixPixel
import lsdb


def test_get_partition_file_paths(small_sky_order1_catalog):
    """Test that we can retrieve file paths for all partitions."""
    file_paths = small_sky_order1_catalog.get_partition_file_paths()
    
    # Check that we get a dictionary
    assert isinstance(file_paths, dict)
    
    # Check that we have paths for all pixels
    pixels = small_sky_order1_catalog.get_healpix_pixels()
    assert len(file_paths) == len(pixels)
    
    # Check that all pixels are present
    for pixel in pixels:
        assert pixel in file_paths
        
    # Check that paths are strings and contain expected patterns
    for pixel, path in file_paths.items():
        assert isinstance(path, str)
        assert "Npix=" in path
        assert str(pixel.pixel) in path
        

def test_get_partition_metadata(small_sky_order1_catalog):
    """Test that we can retrieve detailed metadata for partitions."""
    metadata_df = small_sky_order1_catalog.get_partition_metadata()
    
    # Check that we get a DataFrame
    assert isinstance(metadata_df, pd.DataFrame)
    
    # Check expected columns
    assert "pixel" in metadata_df.columns
    assert "file_path" in metadata_df.columns
    assert "total_size_bytes" in metadata_df.columns
    assert "column_sizes" in metadata_df.columns
    
    # Check that we have metadata for all pixels
    pixels = small_sky_order1_catalog.get_healpix_pixels()
    assert len(metadata_df) == len(pixels)
    
    # Check that file sizes are positive integers
    for _, row in metadata_df.iterrows():
        assert row["total_size_bytes"] > 0
        assert isinstance(row["total_size_bytes"], int)
        
    # Check that column sizes are dictionaries
    for _, row in metadata_df.iterrows():
        assert isinstance(row["column_sizes"], dict)
        assert len(row["column_sizes"]) > 0
        # All column sizes should be positive
        for col_name, col_size in row["column_sizes"].items():
            assert col_size > 0
            assert isinstance(col_name, str)
            

def test_get_memory_estimate_all_columns(small_sky_order1_catalog):
    """Test memory estimation with all columns loaded."""
    estimate = small_sky_order1_catalog.get_memory_estimate()
    
    # Check that we get a dictionary with expected keys
    assert isinstance(estimate, dict)
    assert "total_bytes" in estimate
    assert "total_kb" in estimate
    assert "total_mb" in estimate
    assert "total_gb" in estimate
    assert "per_column_bytes" in estimate
    assert "num_partitions" in estimate
    assert "columns" in estimate
    
    # Check that total bytes is positive
    assert estimate["total_bytes"] > 0
    
    # Check that conversions are correct
    assert estimate["total_kb"] == estimate["total_bytes"] / 1024
    assert estimate["total_mb"] == estimate["total_bytes"] / (1024 ** 2)
    assert estimate["total_gb"] == estimate["total_bytes"] / (1024 ** 3)
    
    # Check that per-column bytes adds up to total
    per_column_sum = sum(estimate["per_column_bytes"].values())
    assert per_column_sum == estimate["total_bytes"]
    
    # Check that num_partitions matches catalog
    assert estimate["num_partitions"] == small_sky_order1_catalog.npartitions
    
    # Check that columns list is populated
    assert len(estimate["columns"]) > 0


def test_get_memory_estimate_subset_columns(small_sky_order1_dir):
    """Test memory estimation with only a subset of columns loaded."""
    # Load catalog with only a subset of columns
    catalog = lsdb.open_catalog(small_sky_order1_dir, columns=["ra", "dec"])
    
    estimate = catalog.get_memory_estimate()
    
    # Check that total bytes is positive but smaller than full catalog
    assert estimate["total_bytes"] > 0
    
    # Check that only the loaded columns are included
    assert "ra" in estimate["columns"]
    assert "dec" in estimate["columns"]
    assert "id" not in estimate["columns"]
    
    # Check that per_column_bytes only has the loaded columns
    assert "ra" in estimate["per_column_bytes"]
    assert "dec" in estimate["per_column_bytes"]
    assert "id" not in estimate["per_column_bytes"]


def test_get_memory_estimate_with_index(small_sky_order1_catalog):
    """Test memory estimation including the index column."""
    estimate_without_index = small_sky_order1_catalog.get_memory_estimate(include_index=False)
    estimate_with_index = small_sky_order1_catalog.get_memory_estimate(include_index=True)
    
    # The estimate with index should be larger
    assert estimate_with_index["total_bytes"] > estimate_without_index["total_bytes"]
    
    # The index column should be included when include_index=True
    index_name = small_sky_order1_catalog._ddf.index.name
    if index_name:
        assert index_name in estimate_with_index["columns"]
        assert index_name not in estimate_without_index["columns"]


def test_get_memory_estimate_after_search(small_sky_order1_catalog):
    """Test memory estimation after filtering with a search."""
    # Perform a cone search that will reduce partitions
    filtered_catalog = small_sky_order1_catalog.cone_search(0, 0, 1)
    
    # The filtered catalog should have fewer partitions
    assert filtered_catalog.npartitions <= small_sky_order1_catalog.npartitions
    
    # Get memory estimate for filtered catalog
    estimate = filtered_catalog.get_memory_estimate()
    
    # Check that estimate is for the filtered partitions
    assert estimate["num_partitions"] == filtered_catalog.npartitions
    
    # If fewer partitions, the estimate should be smaller
    if filtered_catalog.npartitions < small_sky_order1_catalog.npartitions:
        full_estimate = small_sky_order1_catalog.get_memory_estimate()
        assert estimate["total_bytes"] < full_estimate["total_bytes"]


def test_partition_metadata_consistency(small_sky_order1_catalog):
    """Test that partition metadata is consistent across calls."""
    metadata1 = small_sky_order1_catalog.get_partition_metadata()
    metadata2 = small_sky_order1_catalog.get_partition_metadata()
    
    # Should get the same results
    pd.testing.assert_frame_equal(metadata1, metadata2)


def test_file_paths_point_to_existing_pixels(small_sky_order1_catalog):
    """Test that file paths reference the correct pixel numbers."""
    file_paths = small_sky_order1_catalog.get_partition_file_paths()
    
    for pixel, path in file_paths.items():
        # The path should contain the pixel number
        assert f"Npix={pixel.pixel}" in path
        # The path should contain the order
        assert f"Norder={pixel.order}" in path
