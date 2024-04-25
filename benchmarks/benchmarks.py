"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import os

import numpy as np
import pandas as pd

import lsdb
from benchmarks.utils import upsample_array
from lsdb.core.search.box_search import box_filter
from lsdb.core.search.polygon_search import get_cartesian_polygon

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")
DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_ORDER1 = "small_sky_order1"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
BENCH_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_small_sky():
    path = os.path.join(TEST_DIR, DATA_DIR_NAME, SMALL_SKY_DIR_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def load_small_sky_order1():
    path = os.path.join(TEST_DIR, DATA_DIR_NAME, SMALL_SKY_ORDER1)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def load_small_sky_xmatch():
    path = os.path.join(TEST_DIR, DATA_DIR_NAME, SMALL_SKY_XMATCH_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def time_kdtree_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch, require_right_margin=False).compute()


def time_polygon_search():
    """Time polygonal search using sphgeom"""
    small_sky_order1 = load_small_sky_order1().compute()
    # Upsample test catalog to 10,000 points
    catalog_ra = upsample_array(small_sky_order1["ra"].values, 10_000)
    catalog_dec = upsample_array(small_sky_order1["dec"].values, 10_000)
    # Define sky polygon to use in search
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon, _ = get_cartesian_polygon(vertices)
    # Apply vectorized filtering on the catalog points
    polygon.contains(np.radians(catalog_ra), np.radians(catalog_dec))


def time_box_filter_on_partition():
    """Time box search on a single partition"""
    metadata = load_small_sky_order1().hc_structure
    mock_partition_df = pd.DataFrame(
        np.linspace(-1000, 1000, 100_000), columns=[metadata.catalog_info.ra_column]
    )
    box_filter(mock_partition_df, ra=(-20, 40), dec=None, metadata=metadata).compute()


def time_create_midsize_catalog():
    path = os.path.join(BENCH_DATA_DIR, "midsize_catalog")
    return lsdb.read_hipscat(path)


def time_create_large_catalog():
    path = os.path.join(BENCH_DATA_DIR, "large_catalog")
    return lsdb.read_hipscat(path)
