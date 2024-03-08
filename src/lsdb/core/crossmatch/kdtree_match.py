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
    def validate(self, n_neighbors: int = 1, radius_arcsec: float = 1, require_right_margin=True, **kwargs):
        super().validate()
        validate_radius(radius_arcsec)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 1")

        # Check that the margin exists and has a compatible radius.
        if self.right_margin_hc_structure is None:
            if require_right_margin:
                raise ValueError("Right margin is required for cross-match")
        else:
            if self.right_margin_hc_structure.catalog_info.margin_threshold < radius_arcsec:
                raise ValueError("Cross match radius is greater than margin threshold")

    # pylint: disable=unused-argument
    def crossmatch(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """Perform a cross-match between the data from two HEALPix pixels

        Finds the n closest neighbors in the right catalog for each point in the left catalog that
        are within a threshold distance by using a K-D Tree.

        Args:
            n_neighbors (int): The number of neighbors to find within each point
            radius_arcsec (float): The threshold distance in arcseconds beyond which neighbors are not added

        Returns:
            A DataFrame from the left and right tables merged with one row for each pair of
            neighbors found from cross-matching. The resulting table contains the columns from the
            left table with the first suffix appended, the right columns with the second suffix, and
            a "_dist_arcsec" column with the great circle separation between the points.
        """
        # Distance in 3-D space for unit sphere
        radius_degrees = radius_arcsec / 3600.0
        d_chord = 2.0 * math.sin(math.radians(0.5 * radius_degrees))

        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = self._find_crossmatch_indices(
            n_neighbors=n_neighbors, max_distance=d_chord
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
        self, n_neighbors: int, max_distance: float
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
        distances, right_index = tree.query(left_xyz, k=n_neighbors, distance_upper_bound=max_distance)

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
