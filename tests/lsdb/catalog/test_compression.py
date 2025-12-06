import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import lsdb
import inspect
import pyarrow.parquet as pq
import os
from lsdb.catalog.catalog import Catalog


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
        
        converted = small_sky_order1_catalog.to_hats(output_path, catalog_name=catalog_name)

        pqf = pq.ParquetFile(converted)
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # check that defaults are used
        assert any("ZSTD" in str(value) for value in compression_values)


def test_to_hats_custom_compression(small_sky_order1_catalog):
    """Test to_hats custom compression and compression_level are used."""

    # Write to a temporary file with custom compression
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog_name = "test_catalog"
        output_path = Path(tmpdir) / catalog_name
        
        # custom compression parameters
        converted = small_sky_order1_catalog.to_hats(
            output_path,
            catalog_name=catalog_name,
            compression="SNAPPY",
            compression_level=5
        )

        pqf = pq.ParquetFile(converted)
        compression_values = find_compression_values(pqf.metadata.to_dict())

        # check that defaults are used
        assert any("ZSTD" in str(value) for value in compression_values)
