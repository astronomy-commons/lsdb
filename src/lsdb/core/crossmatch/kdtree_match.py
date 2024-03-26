import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_math.validators import validate_radius
from scipy.spatial import KDTree

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm


class KdTreeCrossmatch(AbstractCrossmatchAlgorithm):
    """Nearest neighbor crossmatch using a 3D k-D tree"""

    extra_columns = pd.DataFrame({"_dist_arcsec": pd.Series(dtype=np.dtype("float64"))})

    # pylint: disable=unused-argument,arguments-differ
    def validate(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        min_radius_arcsec: float = 0,
        require_right_margin=True,
        **kwargs,
    ):
        super().validate()
        validate_radius(radius_arcsec)
        if min_radius_arcsec < 0:
            validate_radius(min_radius_arcsec)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 1")
        # Check that the margin exists and has a compatible radius.
        if self.right_margin_hc_structure is None:
            if require_right_margin:
                raise ValueError("Right margin is required for cross-match")
        else:
            if self.right_margin_hc_structure.catalog_info.margin_threshold < radius_arcsec:
                raise ValueError("Cross match radius is greater than margin threshold")
            if self.right_margin_hc_structure.catalog_info.margin_threshold < min_radius_arcsec:
                raise ValueError("Cross match minimum radius is greater than margin threshold")

    # pylint: disable=unused-argument
    def crossmatch(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        min_radius_arcsec: float = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """Perform a cross-match between the data from two HEALPix pixels

        Finds the n closest neighbors in the right catalog for each point in the left catalog that
        are within a threshold distance by using a K-D Tree.

        Args:
            n_neighbors (int): The number of neighbors to find within each point
            radius_arcsec (float): The threshold distance in arcseconds beyond which neighbors are not added
            min_radius_arcsec (float): The minimum distance from which neighbors are added

        Returns:
            A DataFrame from the left and right tables merged with one row for each pair of
            neighbors found from cross-matching. The resulting table contains the columns from the
            left table with the first suffix appended, the right columns with the second suffix, and
            a "_dist_arcsec" column with the great circle separation between the points.
        """
        # Distance in 3-D space for unit sphere
        max_d_chord = _get_chord_distance(radius_arcsec)
        min_d_chord = _get_chord_distance(min_radius_arcsec)

        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = self._find_crossmatch_indices(
            n_neighbors=n_neighbors, min_distance=min_d_chord, max_distance=max_d_chord
        )
        arc_distances = np.degrees(2.0 * np.arcsin(0.5 * chord_distances)) * 3600

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
        extra_columns = pd.DataFrame({"_dist_arcsec": pd.Series(arc_distances, index=out.index)})
        self._append_extra_columns(out, extra_columns)

        return out

    def _find_crossmatch_indices(
        self, n_neighbors: int, min_distance: float, max_distance: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        # calculate the cartesian coordinates of the points
        left_xyz = _lon_lat_to_xyz(
            lon=self.left[self.left_metadata.catalog_info.ra_column].values,
            lat=self.left[self.left_metadata.catalog_info.dec_column].values,
        )
        right_xyz = _lon_lat_to_xyz(
            lon=self.right[self.right_metadata.catalog_info.ra_column].values,
            lat=self.right[self.right_metadata.catalog_info.dec_column].values,
        )

        # Make sure we don't ask for more neighbors than there are points
        n_neighbors = min(n_neighbors, len(right_xyz))

        # construct the KDTree from the right catalog
        tree = KDTree(
            right_xyz,
            leafsize=n_neighbors,
            compact_nodes=True,
            balanced_tree=True,
            copy_data=False,
        )

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
        if n_neighbors > 1:
            left_index = np.stack([left_index] * n_neighbors, axis=1)

        # Infinite distance means no match
        match_mask = np.isfinite(distances)
        return distances[match_mask], left_index[match_mask], right_index[match_mask]


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


def _query_min_max_neighbors(
    tree: KDTree,
    left_xyz: npt.NDArray[np.float64],
    right_xyz: npt.NDArray[np.float64],
    n_neighbors: int,
    min_distance: float,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds `n_neighbors` within a distance range for all points in a pair of partitions"""
    left_tree = KDTree(
        left_xyz, leafsize=n_neighbors, compact_nodes=True, balanced_tree=True, copy_data=False
    )

    # Find the number of neighbors within the minimum distance threshold
    too_close_neighbors = left_tree.query_ball_tree(tree, r=min_distance)
    len_too_close_neighbors = np.asarray([len(neighbors) for neighbors in too_close_neighbors])

    # Make sure we don't ask for more neighbors than there are points
    n_neighbors_to_request = min(n_neighbors + max(len_too_close_neighbors), len(right_xyz))
    distances, right_index = tree.query(left_xyz, k=n_neighbors_to_request, distance_upper_bound=max_distance)

    # Create mask to filter neighbors that are too close
    mask = np.zeros((len(left_xyz), n_neighbors_to_request), dtype=bool)
    for i, how_many_close in enumerate(len_too_close_neighbors):
        remaining = n_neighbors_to_request - how_many_close - n_neighbors
        mask[i] = [False] * how_many_close + [True] * n_neighbors + [False] * remaining

    # Filter points with mask
    final_shape = (len(distances), n_neighbors)
    distances = distances[mask].reshape(final_shape)
    indices = right_index[mask].reshape(final_shape)

    if n_neighbors == 1:
        return distances.flat, indices.flat
    return distances, indices
