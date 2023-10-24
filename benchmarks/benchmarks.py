"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""
import math
import os

import healpy as hp
import numpy as np
import pandas as pd
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from scipy.spatial import KDTree

import lsdb
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm


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


def time_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch).compute()


def time_naive_3dtree_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch, algorithm=Naive3DTreeCrossmatch).compute()


class Naive3DTreeCrossmatch(AbstractCrossmatchAlgorithm):
    """Nearest neighbor crossmatch using a 3-D Tree"""

    def crossmatch(
        self,
        n_neighbors: int = 1,
        d_thresh: float = 0.01,
    ) -> pd.DataFrame:
        # Distance in 3-D space for unit sphere
        d_chord = 2.0 * math.sin(math.radians(0.5 * d_thresh))

        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = self._find_crossmatch_indices(n_neighbors=n_neighbors, distance=d_chord)
        arc_distances = np.degrees(2.0 * np.arcsin(0.5 * chord_distances))

        # rename columns so no same names during merging
        self._rename_columns_with_suffix(self.left, self.suffixes[0])
        self._rename_columns_with_suffix(self.right, self.suffixes[1])

        # concat dataframes together
        self.left.index.name = HIPSCAT_ID_COLUMN
        left_join_part = self.left.iloc[left_idx].reset_index()
        right_join_part = self.right.iloc[right_idx].reset_index(drop=True)
        out = pd.concat(
            [
                left_join_part,
                right_join_part,
            ],
            axis=1,
        )
        out.set_index(HIPSCAT_ID_COLUMN, inplace=True)
        out["_DIST"] = pd.Series(arc_distances, index=out.index)

        return out

    def _find_crossmatch_indices(self, n_neighbors: int, distance: float) -> (np.ndarray[np.float64], np.ndarray[np.int64], np.ndarray[np.int64]):
        # calculate the cartesian coordinates of the points
        left_xyz = _lon_lat_to_xyz(
            self.left[self.left_metadata.catalog_info.dec_column].values,
            self.left[self.left_metadata.catalog_info.ra_column].values,
        )
        right_xyz = _lon_lat_to_xyz(
            self.right[self.right_metadata.catalog_info.dec_column].values,
            self.right[self.right_metadata.catalog_info.ra_column].values,
        )
        n_neighbors = min([n_neighbors, len(right_xyz)])
        # construct the KDTree from the right catalog
        tree = KDTree(right_xyz, leafsize=n_neighbors, compact_nodes=True, balanced_tree=True, copy_data=False)
        # find the indices for the nearest neighbors
        # this is the cross-match calculation
        distances, right_index = tree.query(left_xyz, k=n_neighbors, distance_upper_bound=distance)
        # index of the corresponding row in the left table [[0, 0, 0], [1, 1, 1], [2, 2, 2], ...]
        left_index = np.arange(left_xyz.shape[0])
        # We need make the shape the same as for right_index
        if n_neighbors > 1:
            left_index = np.stack([left_index] * n_neighbors, axis=1)
        # Infinite distance means no match
        match_mask = np.isfinite(distances)
        return distances[match_mask], left_index[match_mask], right_index[match_mask]


def _lon_lat_to_xyz(lon: np.ndarray[np.float64], lat: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """Converts longitude and latitude to cartesian coordinates on the unit sphere

    Args:
        lon (np.ndarray[np.float64]): longitude in radians
        lat (np.ndarray[np.float64]): latitude in radians
    """
    lon = np.radians(lon)
    lat = np.radians(lat)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=1)
