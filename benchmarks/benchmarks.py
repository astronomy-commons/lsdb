"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""
import os

import lsdb
from tests.conftest import test_data_dir, SMALL_SKY_DIR_NAME, SMALL_SKY_XMATCH_NAME


def load_small_sky():
    path = os.path.join(test_data_dir(), SMALL_SKY_DIR_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def load_small_sky_xmatch():
    path = os.path.join(test_data_dir(), SMALL_SKY_XMATCH_NAME)
    return lsdb.read_hipscat(path, catalog_type=lsdb.Catalog)


def time_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch).compute()
