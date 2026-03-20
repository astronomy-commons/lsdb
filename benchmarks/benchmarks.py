"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import tempfile
from pathlib import Path
import os
import sys

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
        cat,
        require_right_margin=False,
        suffixes=("_left", "_right"),
        suffix_method="overlapping_columns",
        log_changes=False,
    )


def time_open_many_columns_list():
    return lsdb.open_catalog(
        BENCH_DATA_DIR / "object_collection",
        columns=["objectId", "coord_dec", "coord_decErr", "coord_ra", "coord_raErr"],
    )


def time_save_big_catalog():
    """Load a catalog with many partitions, and save with write_catalog."""
    mock_partition_df = pd.DataFrame(
        {
            "ra": np.linspace(0, 360, 100_000),
            "dec": np.linspace(-90, 90, 100_000),
            "id": np.arange(100_000, 200_000),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_path:
        kwargs = {
            "catalog_name": "big_sky",
            "catalog_type": "object",
            "lowest_order": 6,
            "highest_order": 10,
            "partition_rows": 500,
        }

        catalog = lsdb.from_dataframe(mock_partition_df, margin_threshold=None, **kwargs)

        catalog.write_catalog(tmp_path)

        read_catalog = hats.read_hats(tmp_path)
        assert len(read_catalog.get_healpix_pixels()) == len(catalog.get_healpix_pixels())

'''
class LSDBDaskGraph:
    """Benchmark dask task graph metrics when opening a catalog"""
    def setup(self):

        # Setup Temp Directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

        # Setup Catalog 1 (Original list is from dask_graph_challenge notebook)
        cat1 = lsdb.generate_catalog(5000,100, lowest_order=1, ra_range=(15.0,25.0), dec_range=(34.0,44.0), seed=1)
        cat1.write_catalog(os.path.join(self.tmp_path, "catalog1"))
        self.cat1 = lsdb.open_catalog(os.path.join(self.tmp_path, "catalog1"))
        # Setup Catalog 2
        cat2 = lsdb.generate_catalog(5000,100, lowest_order=4, ra_range=(15.0,25.0), dec_range=(34.0,44.0), seed=1)
        cat2.write_catalog(os.path.join(self.tmp_path, "catalog2"))
        self.cat2 = lsdb.open_catalog(os.path.join(self.tmp_path, "catalog2"))

        # Open Catalog Graph
        self.open_catalog_cat = self.cat2
        self.open_catalog_graph = self.open_catalog_cat._ddf.optimize().dask

        # Query Catalog Graph
        self.query_cat = self.cat2.query("nested.t>0.3")
        self.query_graph = self.query_cat._ddf.optimize().dask

        # Map_rows Graph
        cat = self.cat2

        def my_sigma(row):
            """map_rows will return a NestedFrame with two columns"""
            return row["nested.flux"] + 1, row["nested.flux"] - 1

        meta = {"plus_one": np.float64, "minus_one": np.float64}
        self.map_cat = cat.map_rows(my_sigma,
                                    columns=["nested.flux"],
                                    output_names=["plus_one", "minus_one"],
                                    meta=meta)
        self.map_graph = self.map_cat._ddf.optimize().dask

        # Partition Selection Graph
        self.select_cat = self.cat2.partitions[2]
        self.select_graph = self.select_cat._ddf.optimize().dask

        # Crossmatch 1 Graph
        self.xmatch1_cat = self.cat1.crossmatch(self.cat2, suffixes=("_l","_r"), suffix_method='all_columns')
        self.xmatch1_graph = self.xmatch1_cat._ddf.optimize().dask

    def teardown(self):
        self.tmp_dir.cleanup()

    # ----- Open Catalog -----
    def track_open_catalog_num_tasks(self):
        graph = self.open_catalog_graph
        return len(graph)

    def track_open_catalog_graph_size(self):
        graph = self.open_catalog_graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_open_catalog_graph_size.unit = "MB"

    #def time_open_catalog(self):
    #    return self.open_catalog_cat.compute()

    #def peakmem_open_catalog(self):
    #    return self.open_catalog_cat.compute()

    # ----- Query -----
    def track_query_num_tasks(self):
        graph = self.query_graph
        return len(graph)

    def track_query_graph_size(self):
        graph = self.query_graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_query_graph_size.unit = "MB"

    #def time_query(self):
    #    return self.query_cat.compute()

    #def peakmem_query(self):
    #    return self.query_cat.compute()

    # ----- map_rows -----
    def track_maprows_num_tasks(self):
        graph = self.map_graph
        return len(graph)

    def track_maprows_graph_size(self):
        graph = self.map_graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_maprows_graph_size.unit = "MB"

    #def time_maprows(self):
    #    return self.map_cat.compute()

    #def peakmem_maprows(self):
    #    return self.map_cat.compute()

    # ----- Partition Selection -----
    def track_select_part_num_tasks(self):
        graph = self.map_graph
        return len(graph)

    def track_select_part_graph_size(self):
        graph = self.map_graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_select_part_graph_size.unit = "MB"

    #def time_select_part(self):
    #    return self.map_cat.compute()

    #def peakmem_select_part(self):
    #    return self.map_cat.compute()

    # ----- Crossmatch 1 -----
    def track_xmatch1_num_tasks(self):
        graph = self.xmatch1_graph
        return len(graph)

    def track_xmatch1_graph_size(self):
        graph = self.xmatch1_graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_xmatch1_graph_size.unit = "MB"

    #def time_xmatch1(self):
    #    return self.xmatch1_cat.compute()

    #def peakmem_xmatch1(self):
    #    return self.xmatch1_cat.compute()
'''


class OpenCatalogDaskGraph:
    """Benchmark dask task graph metrics when opening a catalog"""
    def setup(self):

        # Setup Temp Directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

        # Setup Catalog 2
        cat2 = lsdb.generate_catalog(5000,100, lowest_order=4, ra_range=(15.0,25.0), dec_range=(34.0,44.0), seed=1)
        cat2.write_catalog(os.path.join(self.tmp_path, "catalog2"))
        self.cat2 = lsdb.open_catalog(os.path.join(self.tmp_path, "catalog2"))

        # Open Catalog Graph
        self.cat = self.cat2
        self.graph = self.cat._ddf.optimize().dask

    def teardown(self):
        self.tmp_dir.cleanup()

    # ----- Open Catalog -----
    def track_num_tasks(self):
        graph = self.graph
        return len(graph)

    def track_graph_size(self):
        graph = self.graph
        graph_size = 0
        for key in graph.keys():
            graph_size += sys.getsizeof(graph[key])
        return graph_size/10**6
    track_graph_size.unit = "MB"

    def time_compute(self):
        return self.cat.compute()

    def peakmem_compute(self):
        return self.cat.compute()