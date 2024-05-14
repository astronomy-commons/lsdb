from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_math.validators import validate_radius
from hipscat.pixel_tree import PixelAlignmentType

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_utils import (
    _find_crossmatch_indices,
    _get_arc_separation,
    _get_chord_distance,
    _lon_lat_to_xyz,
)


class KdTreeCrossmatch(AbstractCrossmatchAlgorithm):
    """Nearest neighbor crossmatch using a 3D k-D tree"""

    extra_columns = pd.DataFrame({"_dist_arcsec": pd.Series(dtype=pd.ArrowDtype(pa.float64()))})

    def validate(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        require_right_margin: bool = False,
    ):
        super().validate()
        # Validate radius
        validate_radius(radius_arcsec)
        # Validate number of neighbors
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be greater than 1")
        # Check that the margin exists and has a compatible radius.
        if self.right_margin_hc_structure is None:
            if require_right_margin:
                raise ValueError("Right catalog margin cache is required for cross-match.")
        else:
            if self.right_margin_hc_structure.catalog_info.margin_threshold < radius_arcsec:
                raise ValueError("Cross match radius is greater than margin threshold")
        # Check that the crossmatch strategy has been implemented
        if self.how not in [PixelAlignmentType.INNER, PixelAlignmentType.LEFT]:
            raise NotImplementedError("The cross match strategy must be 'inner' or 'left'")

    def crossmatch(
        self,
        n_neighbors: int = 1,
        radius_arcsec: float = 1,
        # We need it here because the signature is shared with .validate()
        require_right_margin: bool = False,  # pylint: disable=unused-argument
    ) -> pd.DataFrame:
        """Perform a cross-match between the data from two HEALPix pixels

        Finds the n closest neighbors in the right catalog for each point in the left catalog that
        are within a threshold distance by using a K-D Tree.

        Args:
            n_neighbors (int): The number of neighbors to find within each point.
            radius_arcsec (float): The threshold distance in arcseconds beyond which neighbors are not added.

        Returns:
            A DataFrame from the left and right tables merged with one row for each pair of
            neighbors found from cross-matching. The resulting table contains the columns from the
            left table with the first suffix appended, the right columns with the second suffix, and
            a "_dist_arcsec" column with the great circle separation between the points.
        """
        # Distance in 3-D space for unit sphere
        max_d_chord = _get_chord_distance(radius_arcsec)
        # calculate the cartesian coordinates of the points
        left_xyz, right_xyz = self._get_point_coordinates()
        # get matching indices for cross-matched rows
        chord_distances, left_idx, right_idx = _find_crossmatch_indices(
            left_xyz, right_xyz, self.how, n_neighbors, max_d_chord
        )
        return self._create_crossmatch_df(left_idx, right_idx, chord_distances)

    def _get_point_coordinates(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        left_xyz = _lon_lat_to_xyz(
            lon=self.left[self.left_metadata.catalog_info.ra_column].to_numpy(),
            lat=self.left[self.left_metadata.catalog_info.dec_column].to_numpy(),
        )
        right_xyz = _lon_lat_to_xyz(
            lon=self.right[self.right_metadata.catalog_info.ra_column].to_numpy(),
            lat=self.right[self.right_metadata.catalog_info.dec_column].to_numpy(),
        )
        return left_xyz, right_xyz

    def _create_crossmatch_df(
        self,
        left_idx: npt.NDArray[np.int64],
        right_idx: npt.NDArray[np.int64],
        chord_distances: npt.NDArray[np.float64],
    ) -> pd.DataFrame:
        # Rename columns so no same names during merging
        self._rename_columns_with_suffix(self.left, self.suffixes[0])
        self._rename_columns_with_suffix(self.right, self.suffixes[1])
        # Get the rows for the left part of the cross-match
        self.left.index.name = HIPSCAT_ID_COLUMN
        left_join_part = self.left.iloc[left_idx].reset_index()
        # Get the rows for the right part
        right_join_part = self._generate_right_part(right_idx, chord_distances)
        # Concatenate the left and right parts, and set the index.
        out = pd.concat([left_join_part, right_join_part], axis=1)
        out.set_index(HIPSCAT_ID_COLUMN, inplace=True)
        return out

    def _generate_right_part(
        self, right_idx: npt.NDArray[np.int64], chord_distances: npt.NDArray[np.float64]
    ) -> pd.DataFrame:
        """For the right part we need to decide if we keep the non-matches.
        If we do, first we get the rows for the valid matches and set their
        cross-match distance column. Then we fill out the remaining rows for
        the non-matches with empty (nan) data."""

        # First, get the points from the right that are matches. They are matches
        # if the index to the right partition is smaller than the length of the
        # partition and if it is greater than -1.
        match_cond = np.where(np.logical_and(right_idx < len(self.right), right_idx > -1))
        right_join_part = self.right.iloc[right_idx[match_cond]].reset_index(drop=True)

        # For the valid rows, add the extra distance column.
        chord_distances = chord_distances[~np.isinf(chord_distances)]
        arc_distances = _get_arc_separation(chord_distances)
        extra_columns = pd.DataFrame(
            {"_dist_arcsec": pd.Series(arc_distances, dtype=pd.ArrowDtype(pa.float64()))}
        )
        self._append_extra_columns(right_join_part, extra_columns)

        if self.how == PixelAlignmentType.LEFT:
            full_df = pd.DataFrame(None, index=np.arange(len(right_idx)), columns=right_join_part.columns)
            full_df.iloc[match_cond] = right_join_part.to_numpy()
            full_df = full_df.astype(right_join_part.dtypes)
            right_join_part = full_df.reset_index(drop=True)

        return right_join_part
