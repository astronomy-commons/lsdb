from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from hats.pixel_math.validators import validate_radius

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_utils import _find_crossmatch_indices, _get_chord_distance, _lon_lat_to_xyz

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


class KdTreeCrossmatch(AbstractCrossmatchAlgorithm):
    """Nearest neighbor crossmatch using a 3D k-D tree"""

    extra_columns = pd.DataFrame({"_dist_arcsec": pd.Series(dtype=pd.ArrowDtype(pa.float64()))})

    @classmethod
    def validate(
        cls,
        left: Catalog,
        right: Catalog,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
    ):
        super().validate(left, right)
        # Validate radius
        validate_radius(radius_arcsec)
        # Validate number of neighbors
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 1")
        if (
            right.margin is not None
            and right.margin.hc_structure.catalog_info.margin_threshold < radius_arcsec
        ):
            raise ValueError("Cross match radius is greater than margin threshold")

    def perform_crossmatch(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Perform a cross-match between the data from two HEALPix pixels

        Finds the n closest neighbors in the right catalog for each point in the left catalog that
        are within a threshold distance by using a K-D Tree.

        Args:
            n_neighbors (int): The number of neighbors to find within each point.
            radius_arcsec (float): The threshold distance in arcseconds beyond which neighbors are not added

        Returns:
            Indices of the matching rows from the left and right tables found from cross-matching, and a
            datafame with the "_dist_arcsec" column with the great circle separation between the points.
        """
        # Distance in 3-D space for unit sphere
        max_d_chord = _get_chord_distance(radius_arcsec)
        # calculate the cartesian coordinates of the points
        left_xyz, right_xyz = self._get_point_coordinates()
        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = _find_crossmatch_indices(
            left_xyz, right_xyz, n_neighbors=n_neighbors, max_distance=max_d_chord
        )
        arc_distances = np.degrees(2.0 * np.arcsin(0.5 * chord_distances)) * 3600
        extra_columns = pd.DataFrame(
            {"_dist_arcsec": pd.Series(arc_distances, dtype=pd.ArrowDtype(pa.float64()))}
        )
        return left_idx, right_idx, extra_columns

    def _get_point_coordinates(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        left_xyz = _lon_lat_to_xyz(
            lon=self.left[self.left_catalog_info.ra_column].to_numpy(),
            lat=self.left[self.left_catalog_info.dec_column].to_numpy(),
        )
        right_xyz = _lon_lat_to_xyz(
            lon=self.right[self.right_catalog_info.ra_column].to_numpy(),
            lat=self.right[self.right_catalog_info.dec_column].to_numpy(),
        )
        return left_xyz, right_xyz
