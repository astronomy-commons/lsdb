import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import lsdb
import inspect
import pyarrow.parquet as pq
import os
from lsdb.catalog.catalog import Catalog
import hats

# helper function for tests
# find all values associated with the keys "compression" and "compression_level" in pq metadata
def find_compression_values(obj):
    results = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "compression" or key == "compression_level":
                results.append(value)

            # Recurse into the value
            results.extend(find_compression_values(value))

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            results.extend(find_compression_values(item))

    return results


def test_to_hats_default_compression(small_sky_order1_catalog):
    """Test to_hats compression and compression_level defaults are used (ZSTD and 15 respectively)."""

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_name = "test_catalog"
        output_path = Path(tmpdir) / catalog_name
        
        # Call to_hats with default compression (should use ZSTD and level 15)
        # writes files to output_path directory
        small_sky_order1_catalog.to_hats(output_path, catalog_name=catalog_name)
        
        # Check the written parquet files for compression settings
        # Find parquet files in the output directory
        parquet_files = list(output_path.glob('**/*.parquet'))
        assert len(parquet_files) > 0, "No parquet files were written"

        # Filter out metadata files like thumbnail - we want to check actual data files
        data_files = [f for f in parquet_files if 'thumbnail' not in str(f) and 'Npix=' in str(f)]
        assert len(data_files) > 0, "No data parquet files were found"

        # Check compression in the first data parquet file  
        pqf = pq.ParquetFile(data_files[0])
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # Verify that ZSTD compression is used in the data files
        assert any("ZSTD" in str(value) for value in compression_values), "Expected to find ZSTD compression in data files"


def test_to_hats_explicit_zstd_compression(small_sky_order1_catalog):
    """Test to_hats with explicitly specified ZSTD compression."""

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_name = "test_catalog"
        output_path = Path(tmpdir) / catalog_name
        
        # Call to_hats with explicit ZSTD compression
        small_sky_order1_catalog.to_hats(
            output_path, 
            catalog_name=catalog_name,
            compression="ZSTD",
            compression_level=15
        )
        
        # Check the written parquet files for compression settings
        parquet_files = list(output_path.glob('**/*.parquet'))
        assert len(parquet_files) > 0, "No parquet files were written"
        
        # Filter out metadata files like thumbnail - check only data files
        data_files = [f for f in parquet_files if 'thumbnail' not in str(f) and 'Npix=' in str(f)]
        assert len(data_files) > 0, "No data parquet files were found"
        
        # Check compression in the first data parquet file
        pqf = pq.ParquetFile(data_files[0])
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # Check that ZSTD compression is used when explicitly specified
        assert any("ZSTD" in str(value) for value in compression_values), f"Expected ZSTD, found: {compression_values}"


def test_to_hats_custom_compression(small_sky_order1_catalog):
    """Test to_hats custom compression and compression_level are used."""

    # Write to a temporary file with custom compression
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_name = "test_catalog"
        output_path = Path(tmpdir) / catalog_name
        
        # Call to_hats with custom compression parameters (use GZIP which supports levels)
        small_sky_order1_catalog.to_hats(
            output_path,
            catalog_name=catalog_name,
            compression="GZIP",
            compression_level=5
        )

        # Check the written parquet files for compression settings
        parquet_files = list(output_path.glob('**/*.parquet'))
        assert len(parquet_files) > 0, "No parquet files were written"
        
        # Filter out metadata files like thumbnail - check only data files
        data_files = [f for f in parquet_files if 'thumbnail' not in str(f) and 'Npix=' in str(f)]
        assert len(data_files) > 0, "No data parquet files were found"
        
        # Check compression in the first data parquet file
        pqf = pq.ParquetFile(data_files[0])
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # check that custom compression is used
        assert any("GZIP" in str(value) for value in compression_values)


def test_to_hats_snappy_compression(small_sky_order1_catalog):
    """Test to_hats with SNAPPY compression (no compression level)."""

    # Write to a temporary file with SNAPPY compression (which doesn't support levels)
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_name = "test_catalog"
        output_path = Path(tmpdir) / catalog_name

        # Call to_hats with SNAPPY compression
        small_sky_order1_catalog.to_hats(
            output_path,
            catalog_name=catalog_name,
            compression="SNAPPY"
        )

        # Check the written parquet files for compression settings
        parquet_files = list(output_path.glob('**/*.parquet'))
        assert len(parquet_files) > 0, "No parquet files were written"
        
        # Filter out metadata files, check only data files
        data_files = [f for f in parquet_files if 'thumbnail' not in str(f) and 'Npix=' in str(f)]
        assert len(data_files) > 0, "No data parquet files were found"
        
        # Check compression in the first data parquet file
        pqf = pq.ParquetFile(data_files[0])
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # check that SNAPPY compression is used
        assert any("SNAPPY" in str(value) for value in compression_values)
