"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from pathlib import Path

import hats
import numpy as np
import pandas as pd

import lsdb
from benchmarks.utils import upsample_array
from lsdb.core.search.region_search import box_filter, get_cartesian_polygon

TEST_DIR = Path(__file__).parent.parent / "tests"
DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_ORDER1 = "small_sky_order1_collection"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
BENCH_DATA_DIR = Path(__file__).parent / "data"


def load_small_sky():
    return lsdb.open_catalog(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_DIR_NAME)


def load_small_sky_order1():
    return lsdb.open_catalog(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_ORDER1)


def load_small_sky_xmatch():
    return lsdb.open_catalog(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_XMATCH_NAME)


def time_kdtree_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch, require_right_margin=False).compute()


def time_polygon_search():
    """Time polygonal search using sphgeom"""
    small_sky_order1 = load_small_sky_order1().compute()
    # Upsample test catalog to 10,000 points
    catalog_ra = upsample_array(small_sky_order1["ra"].to_numpy(), 10_000)
    catalog_dec = upsample_array(small_sky_order1["dec"].to_numpy(), 10_000)
    # Define sky polygon to use in search
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon = get_cartesian_polygon(vertices)
    # Apply vectorized filtering on the catalog points
    polygon.contains(np.radians(catalog_ra), np.radians(catalog_dec))


def time_box_filter_on_partition():
    """Time box search on a single partition"""
    metadata = load_small_sky_order1().hc_structure
    mock_partition_df = pd.DataFrame(
        {
            metadata.catalog_info.ra_column: np.linspace(-1000, 1000, 100_000),
            metadata.catalog_info.dec_column: np.linspace(-90, 90, 100_000),
        }
    )
    box_filter(mock_partition_df, ra=(-20, 40), dec=(-90, 90), metadata=metadata.catalog_info)


def time_create_midsize_catalog():
    return lsdb.open_catalog(BENCH_DATA_DIR / "midsize_catalog")


def time_create_large_catalog():
    return lsdb.open_catalog(BENCH_DATA_DIR / "large_catalog")


def time_open_many_columns_default():
    return lsdb.open_catalog(BENCH_DATA_DIR / "object_collection")


def time_open_many_columns_all():
    return lsdb.open_catalog(BENCH_DATA_DIR / "object_collection", columns="all")


def time_lazy_crossmatch_many_columns_all_suffixes():
    cat = lsdb.open_catalog(BENCH_DATA_DIR / "object_collection", columns="all")
    return cat.crossmatch(
        cat, require_right_margin=False, suffixes=("_left", "_right"), suffix_method="all_columns"
    )


def time_lazy_crossmatch_many_columns_overlapping_suffixes():
    cat = lsdb.open_catalog(BENCH_DATA_DIR / "object_collection", columns="all")
    return cat.crossmatch(
        cat, require_right_margin=False, suffixes=("_left", "_right"), suffix_method="overlapping_columns"
    )


def time_open_many_columns_list():
    return lsdb.open_catalog(
        BENCH_DATA_DIR / "object_collection",
        columns=["objectId", "coord_dec", "coord_decErr", "coord_ra", "coord_raErr"],
    )


def time_save_big_catalog(tmp_path):
    """Load a catalog with many partitions, and save with to_hats."""
    mock_partition_df = pd.DataFrame(
        {
            "ra": np.linspace(0, 360, 100_000),
            "dec": np.linspace(-90, 90, 100_000),
            "id": np.arange(100_000, 200_000),
        }
    )

    base_catalog_path = tmp_path / "big_sky"

    kwargs = {
        "catalog_name": "big_sky",
        "catalog_type": "object",
        "lowest_order": 6,
        "highest_order": 10,
        "threshold": 500,
    }

    catalog = lsdb.from_dataframe(mock_partition_df, margin_threshold=None, **kwargs)

    catalog.to_hats(base_catalog_path)

    read_catalog = hats.read_hats(base_catalog_path)
    assert len(read_catalog.get_healpix_pixels()) == len(catalog.get_healpix_pixels())
