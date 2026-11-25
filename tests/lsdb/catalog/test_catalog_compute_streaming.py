import dask
import nested_pandas as npd
import pandas as pd
import pytest
from distributed import Client

import lsdb


def local_client():
    return Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None)


def test_catalog_compute_streaming_equals_ddf_compute(small_sky_order1_catalog):
    lsdb_compute = small_sky_order1_catalog.compute(streaming=True).sort_index()
    dask_compute = small_sky_order1_catalog.to_dask_dataframe().compute().sort_index()
    pd.testing.assert_frame_equal(lsdb_compute, dask_compute)


@pytest.mark.parametrize("scheduler", ("threads", "processes", "synchronous", "single-threaded"))
def test_catalog_compute_streaming_with_local_scheduler(small_sky_order1_catalog, scheduler):
    expected = small_sky_order1_catalog.compute()
    with dask.config.set(scheduer=scheduler):
        result = small_sky_order1_catalog.compute(streaming=True).sort_index()
    assert isinstance(result, npd.NestedFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_catalog_compute_streaming_with_distributed_client(small_sky_order1_catalog):
    expected = small_sky_order1_catalog.compute()
    with local_client():
        result = small_sky_order1_catalog.compute(streaming=True).sort_index()
    assert isinstance(result, npd.NestedFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_catalog_compute_streaming_empty_catalog(small_sky_order1_catalog):
    pixel_search = lsdb.PixelSearch.from_radec(80.0, 33.0)
    empty_catalog = small_sky_order1_catalog.search(pixel_search)
    result = empty_catalog.compute(streaming=True).sort_index()
    assert len(result) == 0
    assert list(result.columns) == list(small_sky_order1_catalog.columns)


def tests_catalog_compute_streaming_search(small_sky_order1_catalog):
    # making the search small enough to ensure that at least one chunk
    # processed by the streaming ends up with zero rows.
    search_catalog = small_sky_order1_catalog.box_search(ra=(299, 300), dec=(-40, -39))
    search_df = search_catalog.compute(streaming=True)
    ra_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.ra_column]
    dec_values = search_df[small_sky_order1_catalog.hc_structure.catalog_info.dec_column]
    assert len(search_df) < len(small_sky_order1_catalog.compute())
    assert all(299 <= ra <= 300 for ra in ra_values)
    assert all(-40 <= dec <= -39 for dec in dec_values)
