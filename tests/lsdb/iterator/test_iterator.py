import pytest

import lsdb


def test_catalog_iterator():
    cat = lsdb.generate_catalog(100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1)

    # Test default iteration
    cat_iter = lsdb.CatalogIterator(catalog=cat)
    assert len(cat_iter.partitions_left) == cat.npartitions - cat_iter.partitions_per_chunk
    assert len(cat_iter) == 11

    total_len = 0
    for chunk in cat_iter:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test iter_limit>1
    cat_iter = lsdb.CatalogIterator(catalog=cat, seed=1, iter_limit=5)
    assert len(cat_iter.partitions_left) == cat.npartitions * 5 - cat_iter.partitions_per_chunk

    total_len = 0
    for chunk in cat_iter:
        total_len += len(chunk)
    assert total_len == len(cat) * 5

    # Test chunk size>1
    cat_iter = lsdb.CatalogIterator(catalog=cat, seed=1, partitions_per_chunk=2)
    assert len(cat_iter.partitions_left) == cat.npartitions - 2

    total_len = 0
    for chunk in cat_iter:
        total_len += len(chunk)
    assert total_len == len(cat)

    # Test infinite looping
    cat_iter = lsdb.CatalogIterator(catalog=cat, seed=1, iter_limit=None)

    # Check that we can sample beyond the number of partitions without error
    for _ in range(cat.npartitions * 2):
        chunk = next(cat_iter)


def test_invalid_catalog_input():
    with pytest.raises(
        ValueError, match="The provided catalog input type <class 'str'> is not a lsdb.Catalog object."
    ):
        lsdb.CatalogIterator(catalog="not a catalog")
