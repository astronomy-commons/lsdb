"""
Unit tests for the I/O functionality of the lsdb.nested module.

This module tests the reproducibility of reading and writing Parquet files
using the NestedFrame data structure.
"""

import pandas as pd
import pyarrow as pa

import lsdb.nested as nd


def test_read_parquet(test_dataset, tmp_path):
    """test the reproducibility of read_parquet"""

    # Setup a temporary directory for files
    nested_save_path = tmp_path / "nested"
    test_save_path = tmp_path / "test_dataset"

    # Save Nested to Parquet
    flat_nested = test_dataset.nested.nest.to_flat()
    flat_nested.to_parquet(nested_save_path, write_index=True)

    # Save Base to Parquet
    test_dataset[["a", "b"]].to_parquet(test_save_path, write_index=True)

    # Now read
    base = nd.read_parquet(test_save_path, calculate_divisions=True)
    nested = nd.read_parquet(nested_save_path, calculate_divisions=True)

    # this is read as a large_string, just make it a string
    nested = nested.astype({"band": pd.ArrowDtype(pa.string())})
    base = base.add_nested(nested, "nested")

    # Check the loaded dataset against the original
    assert base.divisions == test_dataset.divisions  # equal divisions
    assert base.compute().equals(test_dataset.compute())  # equal data

    # Check the flat nested datasets
    base_nested_flat = base.nested.nest.to_flat().compute()
    test_nested_flat = base.nested.nest.to_flat().compute()
    assert base_nested_flat.equals(test_nested_flat)
