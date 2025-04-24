"""
Unit tests for the accessor functionality in the lsdb.nested module.

This module tests the behavior and correctness of custom accessors, including
methods like `to_flat`, `to_lists`, and field-specific operations.
"""

import pandas as pd
import pyarrow as pa
import pytest

import lsdb.nested as nd


def test_nest_accessor(test_dataset):
    """test that the nest accessor is correctly tied to columns"""

    # Make sure that nested columns have the accessor available
    assert hasattr(test_dataset.nested, "nest")

    # Make sure we get an attribute error when trying to use the wrong column
    with pytest.raises(AttributeError):
        _ = test_dataset.ra.nest


def test_fields(test_dataset):
    """test the fields accessor property"""
    assert test_dataset.nested.nest.fields == ["t", "flux", "band"]


def test_to_flat():
    """test the to_flat function"""
    nf = nd.datasets.generate_data(10, 100, npartitions=2, seed=1)

    flat_nf = nf.nested.nest.to_flat()

    # check dtypes
    assert flat_nf.dtypes["t"] == pd.ArrowDtype(pa.float64())
    assert flat_nf.dtypes["flux"] == pd.ArrowDtype(pa.float64())
    assert flat_nf.dtypes["band"] == pd.ArrowDtype(pa.string())

    # Make sure we retain all rows
    assert len(flat_nf.loc[1]) == 100

    one_row = flat_nf.compute().iloc[0]

    assert pytest.approx(one_row["t"], 0.01) == 16.0149
    assert pytest.approx(one_row["flux"], 0.01) == 51.2061
    assert one_row["band"] == "r"


def test_to_flat_with_fields():
    """test the to_flat function"""
    nf = nd.datasets.generate_data(10, 100, npartitions=2, seed=1)

    flat_nf = nf.nested.nest.to_flat(fields=["t", "flux"])

    assert "band" not in flat_nf.columns

    # check dtypes
    assert flat_nf.dtypes["t"] == pd.ArrowDtype(pa.float64())
    assert flat_nf.dtypes["flux"] == pd.ArrowDtype(pa.float64())

    # Make sure we retain all rows
    assert len(flat_nf.loc[1]) == 100

    one_row = flat_nf.compute().iloc[0]

    assert pytest.approx(one_row["t"], 0.01) == 16.0149
    assert pytest.approx(one_row["flux"], 0.01) == 51.2061


def test_to_lists():
    """test the to_lists function"""

    nf = nd.datasets.generate_data(10, 100, npartitions=2, seed=1)
    list_nf = nf.nested.nest.to_lists()

    # check dtypes
    assert list_nf.dtypes["t"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_nf.dtypes["flux"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_nf.dtypes["band"] == pd.ArrowDtype(pa.list_(pa.string()))

    # Make sure we have a single row for an id
    assert len(list_nf.loc[1]) == 1

    # Make sure we retain all rows -- double loc for speed and pandas get_item
    assert len(list_nf.loc[1].compute().loc[1]["t"]) == 100

    one_row = list_nf.compute().iloc[1]
    # spot-check values
    assert pytest.approx(one_row["t"][0], 0.01) == 19.3652
    assert pytest.approx(one_row["flux"][0], 0.01) == 61.7461
    assert one_row["band"][0] == "g"


def test_to_lists_with_fields():
    """test the to_lists function"""
    nf = nd.datasets.generate_data(10, 100, npartitions=2, seed=1)
    list_nf = nf.nested.nest.to_lists(fields=["t", "flux"])

    assert "band" not in list_nf.columns

    # check dtypes
    assert list_nf.dtypes["t"] == pd.ArrowDtype(pa.list_(pa.float64()))
    assert list_nf.dtypes["flux"] == pd.ArrowDtype(pa.list_(pa.float64()))

    # Make sure we have a single row for an id
    assert len(list_nf.loc[1]) == 1

    # Make sure we retain all rows -- double loc for speed and pandas get_item
    assert len(list_nf.loc[1].compute().loc[1]["t"]) == 100

    one_row = list_nf.compute().iloc[1]
    # spot-check values
    assert pytest.approx(one_row["t"][0], 0.01) == 19.3652
    assert pytest.approx(one_row["flux"][0], 0.01) == 61.7461
