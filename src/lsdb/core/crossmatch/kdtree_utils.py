import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


def _find_crossmatch_indices(
    left_xyz: npt.NDArray[np.float64],
    right_xyz: npt.NDArray[np.float64],
    n_neighbors: int,
    max_distance: float,
    min_distance: float = 0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # Make sure we don't ask for more neighbors than there are points
    n_neighbors = min(n_neighbors, len(right_xyz))

    # construct the KDTree from the right catalog
    tree = KDTree(right_xyz, compact_nodes=True, balanced_tree=True, copy_data=False)

    # find the indices for the nearest neighbors
    # this is the cross-match calculation
    distances, right_index = (
        _query_min_max_neighbors(tree, left_xyz, right_xyz, n_neighbors, min_distance, max_distance)
        if min_distance > 0
        else tree.query(left_xyz, k=n_neighbors, distance_upper_bound=max_distance)
    )

    # index of the corresponding row in the left table [[0, 0, 0], [1, 1, 1], [2, 2, 2], ...]
    left_index = np.arange(left_xyz.shape[0])

    # We need make the shape the same as for right_index
    if n_neighbors > 1 or min_distance > 0:
        left_index = np.stack([left_index] * n_neighbors, axis=1)

    # Infinite distance means no match
    match_mask = np.isfinite(distances)
    return distances[match_mask], left_index[match_mask], right_index[match_mask]


# pylint: disable=too-many-locals
def _query_min_max_neighbors(
    tree: KDTree,
    left_xyz: npt.NDArray[np.float64],
    right_xyz: npt.NDArray[np.float64],
    n_neighbors: int,
    min_distance: float,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds `n_neighbors` within a distance range for all points in a pair of partitions"""
    left_tree = KDTree(left_xyz, compact_nodes=True, balanced_tree=True, copy_data=False)

    # Find the number of neighbors within the minimum distance threshold.
    len_too_close_neighbors = np.zeros(left_xyz.shape[0], dtype=np.int64)
    # The sparse distance matrix is a dictionary of keys. It contains the pairs of neighbors
    # and their respective distances, in the form of [i,j,distance], i and j being the indices
    # of the left and right partitions, respectively. Accessing "i" we obtain all the indices
    # of the points in the left partition with a match in the right partition, for the specified
    # minimum distance. These are the neighbors that are too close.
    left_indices_too_close = left_tree.sparse_distance_matrix(
        tree, max_distance=min_distance, output_type="ndarray"
    )["i"]
    unique, counts = np.unique(left_indices_too_close, return_counts=True)
    len_too_close_neighbors[unique] = counts

    # Make sure we don't ask for more neighbors than there are points.
    n_neighbors_to_request = min(n_neighbors + max(len_too_close_neighbors), len(right_xyz))

    distances, right_index = tree.query(left_xyz, k=n_neighbors_to_request, distance_upper_bound=max_distance)
    if n_neighbors_to_request == 1:
        distances = np.array(
            [
                distances,
            ]
        ).T
        right_index = np.array(
            [
                right_index,
            ]
        ).T

    # Create mask to filter neighbors that are too close. First we start with all false.
    mask = np.zeros((len(left_xyz), n_neighbors_to_request), dtype=bool)

    # Compute the indices of the mask that should be true e.g. for the mask [[0,0,1,1], [0,1,1,0], [1,1,0,0]]
    # The indices of the 1s are [[0,2],[0,3],[1,1],[1,2],[2,0],[2,1]]
    # Except numpy uses the transverse of these indices, so we want [[0,0,1,1,2,2], [2,3,1,2,0,1]]

    # The first array for the indices is [0 * n_neighbors, 1 * n_neighbors, ..., len(left) * n_neighbors]
    mask_ones_0 = np.repeat(np.arange(len_too_close_neighbors.shape[0]), n_neighbors)
    # The second array of the indices is [len_too_close[0], len_too_close[0] + 1, ... len_too_close[0]
    # + n_neighbors, repeated for each len_too_close]
    mask_ones_1 = np.tile(np.arange(n_neighbors), len_too_close_neighbors.shape[0]) + np.repeat(
        len_too_close_neighbors, n_neighbors
    )
    # If there are fewer points than those requested, clip indices.
    if n_neighbors + max(len_too_close_neighbors) > len(right_xyz):
        mask_ones_1 = np.clip(mask_ones_1, None, len(right_xyz) - 1)

    # Set the mask to one for the indices it should be one.
    mask[mask_ones_0, mask_ones_1] = True

    # We need to generate a final mask (out_mask) to remove all the leading False(s) from the first mask.
    # These leading False(s) appear for points that have matches closer than the minimum distance radius.
    out_mask = np.zeros((len(left_xyz), n_neighbors), dtype=bool)
    out_mask_ones_1: np.ndarray = mask_ones_1 - np.repeat(len_too_close_neighbors, n_neighbors)
    out_mask[mask_ones_0, out_mask_ones_1] = True

    # These arrays will hold the final distances and respective indices for the specified number
    # of neighbors. By default, they are initialized with values for non-match, i.e., distances
    # are filled with infinite values and indices with an index that corresponds to the number of
    # points in the right partition.
    out_distances = np.full((len(left_xyz), n_neighbors), fill_value=np.inf)
    out_indexes = np.full((len(left_xyz), n_neighbors), fill_value=len(right_xyz), dtype=np.int64)

    # Finally, apply mask to filter points.
    out_indexes[out_mask] = right_index[mask]
    out_distances[out_mask] = distances[mask]

    return out_distances, out_indexes


def _lon_lat_to_xyz(lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


def _get_chord_distance(radius_arcsec: float) -> float:
    """Calculates the distance between two points on the surface of the unit sphere,
    for a given radius, in arcseconds

    Args:
        radius_arcsec (float): Tbe radius, in arcseconds

    Returns:
        The chord distance between the two points on the unit sphere.
    """
    radius_degrees = radius_arcsec / 3600.0
    return 2.0 * math.sin(math.radians(0.5 * radius_degrees))
