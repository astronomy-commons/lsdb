import healpy as hp
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm


class KdTreeCrossmatch(AbstractCrossmatchAlgorithm):
    """Nearest neighbor crossmatch using a K-D Tree"""

    def crossmatch(
        self,
        n_neighbors: int = 1,
        d_thresh: float = 0.01,
    ) -> pd.DataFrame:
        """Perform a cross-match between the data from two HEALPix pixels

        Finds the n closest neighbors in the right catalog for each point in the left catalog that
        are within a threshold distance by using a K-D Tree.

        Args:
            n_neighbors (int): The number of neighbors to find within each point
            d_thresh (float): The threshold distance in degrees beyond which neighbors are not added

        Returns:
            A DataFrame from the left and right tables merged with one row for each pair of
            neighbors found from cross-matching. The resulting table contains the columns from the
            left table with the first suffix appended, the right columns with the second suffix, and
            a column with the name {AbstractCrossmatchAlgorithm.DISTANCE_COLUMN_NAME} with the
            great circle separation between the points.
        """

        # get matching indices for cross-matched rows
        left_idx, right_idx = self._find_crossmatch_indices(n_neighbors)

        # filter indexes to only include rows with points within the distance threshold
        (
            distances,
            left_ids_filtered,
            right_ids_filtered,
        ) = self._filter_indexes_to_threshold(left_idx, right_idx, d_thresh)

        # rename columns so no same names during merging
        self._rename_columns_with_suffix(self.left, self.suffixes[0])
        self._rename_columns_with_suffix(self.right, self.suffixes[1])

        # concat dataframes together
        self.left.index.name = "_hipscat_index"
        left_join_part = self.left.iloc[left_ids_filtered].reset_index()
        right_join_part = self.right.iloc[right_ids_filtered].reset_index(drop=True)
        out = pd.concat(
            [
                left_join_part,
                right_join_part,
            ],
            axis=1,
        )
        out.set_index("_hipscat_index", inplace=True)
        out["_DIST"] = distances

        return out

    def _find_crossmatch_indices(self, n_neighbors):
        # calculate the gnomic distances to use with the KDTree
        clon, clat = hp.pix2ang(hp.order2nside(self.left_order), self.left_pixel, nest=True, lonlat=True)
        xy1 = _frame_gnomonic(self.left, self.left_metadata.catalog_info, clon, clat)
        xy2 = _frame_gnomonic(self.right, self.right_metadata.catalog_info, clon, clat)
        # construct the KDTree from the right catalog
        tree = KDTree(xy2, leaf_size=2)
        # find the indices for the nearest neighbors
        # this is the cross-match calculation
        _, inds = tree.query(xy1, k=min([n_neighbors, len(xy2)]))
        # numpy indexing to join the two catalogs
        # index of each row in the output table # (0... number of output rows)
        out_idx = np.arange(len(self.left) * n_neighbors)
        # index of the corresponding row in the left table (0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
        left_idx = out_idx // n_neighbors
        # index of the corresponding row in the right table (22, 33, 44, 55, 66, ...)
        right_idx = inds.ravel()
        return left_idx, right_idx

    def _filter_indexes_to_threshold(self, left_idx, right_idx, d_thresh):
        """
        Filters indexes to merge dataframes to the points separated by distances within the
        threshold

        Returns:
            A tuple of (distances, filtered_left_indices, filtered_right_indices)
        """
        left_catalog_info = self.left_metadata.catalog_info
        right_catalog_info = self.right_metadata.catalog_info
        # align radec to indices
        left_radec = self.left[[left_catalog_info.ra_column, left_catalog_info.dec_column]]
        left_radec_aligned = left_radec.iloc[left_idx]
        right_radec = self.right[[right_catalog_info.ra_column, right_catalog_info.dec_column]]
        right_radec_aligned = right_radec.iloc[right_idx]

        # store the indices from each row
        distances_df = pd.DataFrame.from_dict({"_left_idx": left_idx, "_right_idx": right_idx})

        # calculate distances of each pair
        distances_df[self.DISTANCE_COLUMN_NAME] = _great_circle_dist(
            left_radec_aligned[left_catalog_info.ra_column].values,
            left_radec_aligned[left_catalog_info.dec_column].values,
            right_radec_aligned[right_catalog_info.ra_column].values,
            right_radec_aligned[right_catalog_info.dec_column].values,
        )
        # cull based on the distance threshold
        distances_df = distances_df.loc[distances_df[self.DISTANCE_COLUMN_NAME] < d_thresh]
        left_ids_filtered = distances_df["_left_idx"]
        right_ids_filtered = distances_df["_right_idx"]
        distances = distances_df[self.DISTANCE_COLUMN_NAME].to_numpy()
        return distances, left_ids_filtered, right_ids_filtered


def _great_circle_dist(lon1, lat1, lon2, lat2):
    """
    function that calculates the distance between two points
        p1 (lon1, lat1) or (ra1, dec1)
        p2 (lon2, lat2) or (ra2, dec2)

        can be np.array()
        returns np.array()
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    return np.degrees(
        2
        * np.arcsin(
            np.sqrt(
                (np.sin((lat1 - lat2) * 0.5)) ** 2
                + np.cos(lat1) * np.cos(lat2) * (np.sin((lon1 - lon2) * 0.5)) ** 2
            )
        )
    )


def _frame_gnomonic(data_frame, catalog_info, clon, clat):
    """
    method taken from lsd1:
    creates a np.array of gnomonic distances for each source in the dataframe
    from the center of the ordered pixel. These values are passed into
    the kdtree NN query during the xmach routine.
    """
    phi = np.radians(data_frame[catalog_info.dec_column].values)
    lam = np.radians(data_frame[catalog_info.ra_column].values)
    phi1 = np.radians(clat)
    lam0 = np.radians(clon)

    cosc = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    x_projected = np.cos(phi) * np.sin(lam - lam0) / cosc
    y_projected = (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0)) / cosc

    ret = np.column_stack((np.degrees(x_projected), np.degrees(y_projected)))
    del phi, lam, phi1, lam0, cosc, x_projected, y_projected
    return ret
