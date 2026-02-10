import numpy as np
import pytest

import lsdb
from lsdb.streams import CatalogStream, InfiniteStream


def test_catalog_stream():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)

    # Test default iteration
    cat_stream = CatalogStream(catalog=cat)

    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk
    assert len(cat_iter) == 11

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test chunk size>1
    cat_stream = CatalogStream(catalog=cat, seed=1, partitions_per_chunk=2)
    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test shuffing=False
    cat_stream = CatalogStream(catalog=cat, seed=1, shuffle=False)
    cat_iter = iter(cat_stream)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_stream.partitions_per_chunk
    assert len(cat_iter) == 11

    total_len = 0
    for chunk in cat_stream:
        total_len += len(chunk)
    assert total_len == len(cat)


def test_infinite_stream():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    # Test infinite looping
    cat_stream = InfiniteStream(catalog=cat, seed=1)
    cat_iter = iter(cat_stream)

    # Check that we can sample beyond the number of partitions without error
    for _ in range(cat.npartitions * 2):
        next(cat_iter)


def test_invalid_catalog_input():
    with pytest.raises(
        ValueError, match="The provided catalog input type <class 'str'> is not a lsdb.Catalog object."
    ):
        CatalogStream(catalog="not a catalog")


def test_rng_split():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)
    cat_stream = CatalogStream(catalog=cat, seed=1)
    cat_iter = iter(cat_stream)

    # Check that the RNG is properly split and produces different sequences for different iterations
    first_partitions = cat_iter.partitions_left.copy()
    next(cat_iter)  # Advance the iterator to change the RNG state
    second_partitions = cat_iter.partitions_left.copy()

    assert not np.array_equal(
        first_partitions, second_partitions
    ), "RNG should produce different sequences after splitting."
