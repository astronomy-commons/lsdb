import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pytest

import lsdb
from lsdb.streams import CatalogStream, InfiniteStream
from lsdb.streams.catalog_streams import CrossMatchStream


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

    # Check that length raises an error for InfiniteStream
    with pytest.raises(TypeError, match="Length is not defined for an InfiniteStream."):
        len(cat_iter)


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


def test_stream_from_search_filter():
    """test to make sure pixel set handoff from a search filter works"""
    # explicit test for https://github.com/astronomy-commons/lsdb/issues/1549

    cat = lsdb.generate_catalog(1000, 2, seed=1, ra_range=(0.0, 10.0), dec_range=(0, 10.0), partition_rows=20)
    cat = cat.cone_search(ra=0, dec=0, radius_arcsec=1000)

    # If this runs without error, then the stream is working
    for i, chunk in enumerate(CatalogStream(cat)):
        if i > 0:
            break
        print(chunk)


def _crossmatch_stream_catalogs(monkeypatch):
    """Two small overlapping catalogs for CrossMatchStream tests.

    CrossMatchStream reads each catalog's on-disk point-map skymap to
    estimate match fractions; in-memory catalogs have none, so serve a
    synthetic all-ones skymap (every pixel estimated to match fully)."""
    monkeypatch.setattr(
        "lsdb.streams.catalog_streams.read_skymap",
        lambda hc_catalog, path: np.ones(hp.order2npix(5)),
    )
    left_df = pd.DataFrame(
        {
            "ra": np.linspace(15.0, 25.0, 64),
            "dec": np.linspace(34.0, 44.0, 64),
            "id": np.arange(64),
        }
    )
    right_df = pd.DataFrame(
        {
            "ra": np.linspace(15.0, 25.0, 32),
            "dec": np.linspace(34.0, 44.0, 32),
            "id": np.arange(32),
        }
    )
    left = lsdb.from_dataframe(
        left_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=4,
        highest_order=5,
        margin_threshold=30,
    )
    right = lsdb.from_dataframe(
        right_df,
        ra_column="ra",
        dec_column="dec",
        lowest_order=4,
        highest_order=5,
        margin_threshold=30,
    )
    return left, right


def test_crossmatch_stream_yields_left_joined_rows(monkeypatch):
    left, right = _crossmatch_stream_catalogs(monkeypatch)
    stream = CrossMatchStream(
        left,
        {"other": right},
        client=None,
        partitions_per_chunk=2,
        seed=1,
        count_fraction_threshold=0.0,
    )
    cat_iter = iter(stream)
    chunk = next(cat_iter)
    assert len(chunk) > 0
    # never-skip threshold: every row keeps the right catalog's columns,
    # suffixed with the right catalog's name
    assert f"id_{right.name}" in chunk.columns
    assert len(chunk["id"]) == chunk["id"].notna().sum()


def test_stream_submits_next_chunk_before_waiting_for_result():
    """The next chunk is submitted before the current result is awaited.

    Graph construction and scheduling for chunk N+1 must overlap chunk N's
    computation (the class docstring's pre-fetch claim), not run on the
    consumer's critical path after it. Ordering is observable by logging
    submissions and result reads; with a real dask client this is what
    hides CrossMatchStream's per-pixel graph construction behind fetches."""
    cat = lsdb.generate_catalog(
        100, 2, lowest_order=4, ra_range=(15.0, 25.0), dec_range=(34.0, 44.0), seed=1
    )
    events = []

    class _LoggingFuture:
        def __init__(self, inner):
            self.inner = inner

        def result(self):
            value = self.inner.result()
            events.append("read")
            return value

    class RecordingStream(CatalogStream):
        def submit_next_partitions(self, partitions):
            future = super().submit_next_partitions(partitions)
            events.append(f"submit:{len(partitions)}")
            return _LoggingFuture(future)

    stream = RecordingStream(catalog=cat, seed=1, partitions_per_chunk=2)
    cat_iter = iter(stream)
    assert events == ["submit:2"]  # the first chunk is pre-submitted

    next(cat_iter)
    # the second chunk was submitted BEFORE the first result was read
    assert events == ["submit:2", "submit:2", "read"]

    # iteration still yields every row exactly once
    total = 0
    for chunk in RecordingStream(catalog=cat, seed=1, partitions_per_chunk=2):
        total += len(chunk)
    assert total == len(cat)
