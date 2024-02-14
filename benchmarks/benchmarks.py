"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import os

import numpy as np

import lsdb
from benchmarks.utils import upsample_array
from lsdb.core.search.polygon_search import get_cartesian_polygon

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")
DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_ORDER1 = "small_sky_order1"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"


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
    small_sky.crossmatch(small_sky_xmatch).compute()


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
