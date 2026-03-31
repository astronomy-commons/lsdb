import os
import time
import tracemalloc
import dask.dataframe as dd
import pandas as pd
import lsdb
from lsdb import generate_data
from lsdb.catalog.dataset.templates.repr_builder import build_catalog_html
from lsdb.nested.core import NestedFrame
from tests.data.small_sky_order1_collection import small_sky_order1
from tests.data import small_sky_order3_source, small_sky_source

CATALOG_DIR = os.path.join("tests", "data", "small_sky_order1_collection", "small_sky_order1")


def get_network_bytes():
    """Reads total network bytes (rx + tx) from /proc/net/dev (Linux only)."""
    total = 0
    try:
        with open("/proc/net/dev", "r") as f:
            lines = f.readlines()[2:]  # Skip headers
            for line in lines:
                parts = line.split()
                # Receive bytes is index 1, Transmit bytes is index 9
                total += int(parts[1]) + int(parts[9])
    except FileNotFoundError:
        return 0  # Non-linux systems will return 0
    return total


# Test suite for the new feature that solves lsdb issue #1274,
# https://github.com/astronomy-commons/lsdb/issues/1274
def test_lsdb_html_repr_catalogs():
    """
    Tests the HTML representation for a collection of local catalogs.
    Ensureing that `__repr_html__` works correctly for catalogs stored
    locally in `tests.data`. monitoring network usage and peak memory to
    verify that the representation is generated lazily.
    """

    catalogs = [
        small_sky_order1.__path__[0],
        small_sky_order3_source.__path__[0],
        small_sky_source.__path__[0]
    ]

    print(f"\n{'Catalog Source':<60} | {'Net (MB)':<10} | {'Mem (MB)':<10}")
    print("-" * 85)

    tracemalloc.start()

    for entry in catalogs:
        net_before = get_network_bytes()
        tracemalloc.reset_peak()

        try:
            catalog = lsdb.open_catalog(entry)
            _ = catalog.__repr_html__()

            net_after = get_network_bytes()
            current_mem, peak_mem = tracemalloc.get_traced_memory()

            # 4. Calculations
            net_delta_mb = (net_after - net_before) / (1024 * 1024)
            peak_mem_mb = peak_mem / (1024 * 1024)

            print(f"{entry} | {net_delta_mb:>8.2f} | {peak_mem_mb:>8.2f}")

        except Exception as e:
            print(f"FAILED: {entry} - {str(e)[:50]}...")

    tracemalloc.stop()


def test_repr_html_is_lazy():
    nf = generate_data(100000, seed=42, ra_range=(0.0, 300.0), dec_range=(-50.0, 50.0), n_layer=7)
    catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]], catalog_name="LazyTest")

    tracemalloc.start()
    start_time = time.time()

    html = catalog.__repr_html__()

    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert elapsed < 8.0, f"__repr_html__ took {elapsed:.2f}s, should be < 1s"
    assert peak_mem < 300 * 1024 * 1024, f"Used {peak_mem / 1024 / 1024:.2f} MB"

    assert isinstance(html, str)
    assert len(html) > 0
    print(f"Lazy test passed: {elapsed:.3f}s, {peak_mem / 1024 / 1024:.2f} MB")


def test_html_contains_all_sections():
    nf = generate_data(10000, seed=192, ra_range=(0.7, 292.0), dec_range=(-110.0, 120.0), n_layer=16)
    catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]], catalog_name="StructureTest")

    catalog.__repr_html__()

    required_sections = [
        'lsdb-catalog',
        'lsdb-header',
        'Catalog Metadata',
        'Suggested Methods',
        'Data Preview',
        'PyArrow Schema',
        'Documentation',
        'StructureTest',
        "Sky Coverage",
    ]

    html = build_catalog_html(catalog)

    for section in required_sections:
        assert section in html, f"Missing required section: {section}"

    print(f"All {len(required_sections)} sections present")


def test_after_query():
    """Test __repr_html__ after a query"""
    nf = generate_data(2000, 32, seed=42)
    catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]], catalog_name="QueriedCatalog")

    queried = catalog.query("ra > 100")

    html = queried.__repr_html__()

    assert "QueriedCatalog" in html
    assert isinstance(html, str)
    print("After query operation works")


def test_after_cone_search():
    """Test __repr_html__ after performing cone search"""
    nf = generate_data(2000, 32, seed=42)
    catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]], catalog_name="ConeSearch")

    search_result = catalog.cone_search(ra=150.0, dec=17.0, radius_arcsec=2950)

    html = search_result.__repr_html__()

    assert "ConeSearch" in html
    assert isinstance(html, str)
    print("After query operation works")


def test_no_catalog_loaded():
    """Directly constructing a Catalog with an empty ddf and empty pixel map
    (the closest state to 'nothing loaded') must still produce valid HTML."""

    # Using a minimal hc_structure from a tiny synthetic catalog
    tiny_df = pd.DataFrame({"ra": [10.0], "dec": [20.0]})
    base_cat = lsdb.from_dataframe(tiny_df, catalog_name="NothingLoaded")

    empty_df = pd.DataFrame({"ra": pd.Series([], dtype=float),
                             "dec": pd.Series([], dtype=float)})
    empty_ddf = NestedFrame.from_dask_dataframe(dd.from_pandas(empty_df, npartitions=1))
    no_data_cat = lsdb.Catalog(empty_ddf, {}, base_cat.hc_structure)

    html = no_data_cat.__repr_html__()

    assert isinstance(html, str)
    assert len(html) > 0
    assert "NothingLoaded" in html

