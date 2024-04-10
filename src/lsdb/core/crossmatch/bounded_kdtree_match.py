from __future__ import annotations

import numpy as np
import pandas as pd

from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_utils import _find_crossmatch_indices, _get_chord_distance


class BoundedKdTreeCrossmatch(KdTreeCrossmatch):
    """Nearest neighbor crossmatch using a distance range"""

    def validate(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        require_right_margin: bool = True,
        min_radius_arcsec: float = 0,
        **kwargs,
    ):
        super().validate(n_neighbors, radius_arcsec, require_right_margin)
        if min_radius_arcsec < 0:
            raise ValueError("The minimum radius must be non-negative")
        if radius_arcsec <= min_radius_arcsec:
            raise ValueError("Cross match maximum radius must be greater than cross match minimum radius")

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
            n_neighbors (int): The number of neighbors to find within each point.
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
        # calculate the cartesian coordinates of the points
        left_xyz, right_xyz = self._get_point_coordinates()
        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = _find_crossmatch_indices(
            left_xyz, right_xyz, n_neighbors=n_neighbors, min_distance=min_d_chord, max_distance=max_d_chord
        )
        arc_distances = np.degrees(2.0 * np.arcsin(0.5 * chord_distances)) * 3600
        return self._create_crossmatch_df(left_idx, right_idx, arc_distances)
