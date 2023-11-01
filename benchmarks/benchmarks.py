"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""
import os

import lsdb

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")
DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"


def load_small_sky():
    path = os.path.join(TEST_DIR, DATA_DIR_NAME, SMALL_SKY_DIR_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def load_small_sky_xmatch():
    path = os.path.join(TEST_DIR, DATA_DIR_NAME, SMALL_SKY_XMATCH_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def time_kdtree_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch).compute()
