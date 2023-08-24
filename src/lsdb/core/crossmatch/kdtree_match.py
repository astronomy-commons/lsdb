from typing import Tuple

import numpy as np
import pandas as pd
import healpy as hp
import hipscat as hc
from sklearn.neighbors import KDTree


def kd_tree_crossmatch(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_order: int,
    left_pixel: int,
    right_order: int,
    right_pixel: int,
    left_metadata: hc.catalog.Catalog,
    right_metadata: hc.catalog.Catalog,
    suffixes: Tuple[str, str],
    n_neighbors: int = 1,
    d_thresh: float = 0.01,
) -> pd.DataFrame:
    """Perform a cross-match between the data from two HEALPix pixels

    Finds the n closest neighbors in the right catalog for each point in the left catalog that
    are within a threshold distance by using a K-D Tree.

    Args:
        left (pd.DataFrame): Data from the pixel in the left tree
        right (pd.DataFrame): Data from the pixel in the right tree
        left_order (int): The HEALPix order of the left pixel
        left_pixel (int): The HEALPix pixel number in NESTED ordering of the left pixel
        right_order (int): The HEALPix order of the right pixel
        right_pixel (int): The HEALPix pixel number in NESTED ordering of the right pixel
        left_metadata (hipscat.Catalog): The hipscat Catalog object with the metadata of the left
            catalog
        right_metadata (hipscat.Catalog): The hipscat Catalog object with the metadata of the right
            catalog
        suffixes (Tuple[str,str]): A pair of suffixes to be appended to the end of each column name,
            with the first appended to the left columns and the second to the right columns
        n_neighbors (int): The number of neighbors to find within each point
        d_thresh (float): The threshold distance beyond which neighbors are not added

    Returns:
        A DataFrame from the left and right tables merged with one row for each pair of neighbors
        found from cross-matching. The resulting table contains the columns from the left table with
        the first suffix appended, the right columns with the second suffix, and a `_DIST` column
        with the great circle separation between the points.
    """

    left = left.copy(deep=False)
    right = right.copy(deep=False)

    left_idx, right_idx = _find_crossmatch_indices(left, left_metadata, right, right_metadata,
                                                   left_order, left_pixel, n_neighbors)

    # filter indexes to only include rows with points within the distance threshold
    distances, left_ids_filtered, right_ids_filtered = _filter_indexes_to_threshold(
        d_thresh, left, left_idx, left_metadata.catalog_info, right, right_idx, right_metadata.catalog_info
    )

    # rename columns so no same names during merging
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left.rename(columns=left_columns_renamed, inplace=True)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right.rename(columns=right_columns_renamed, inplace=True)

    # concat dataframes together
    left.index.name = "_hipscat_index"
    left_join_part = left.iloc[left_ids_filtered].reset_index()
    right_join_part = right.iloc[right_ids_filtered].reset_index(drop=True)
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


def _find_crossmatch_indices(left, left_metadata, right, right_metadata, order, pixel, n_neighbors):
    # calculate the gnomic distances to use with the KDTree
    clon, clat = hp.pix2ang(
        hp.order2nside(order), pixel, nest=True, lonlat=True
    )
    xy1 = frame_gnomonic(left, left_metadata.catalog_info, clon, clat)
    xy2 = frame_gnomonic(right, right_metadata.catalog_info, clon, clat)
    # construct the KDTree from the right catalog
    tree = KDTree(xy2, leaf_size=2)
    # find the indices for the nearest neighbors
    # this is the cross-match calculation
    _, inds = tree.query(xy1, k=min([n_neighbors, len(xy2)]))
    # numpy indexing to join the two catalogs
    # index of each row in the output table # (0... number of output rows)
    out_idx = np.arange(len(left) * n_neighbors)
    # index of the corresponding row in the left table (0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
    left_idx = out_idx // n_neighbors
    # index of the corresponding row in the right table (22, 33, 44, 55, 66, ...)
    right_idx = inds.ravel()
    return left_idx, right_idx


def _filter_indexes_to_threshold(
    d_thresh, left, left_idx, left_catalog_info, right, right_idx, right_catalog_info
):
    """
    Filters indexes to merge dataframes to the points separated by distances within the threshold

    Returns:
        A tuple of (distances, filtered_left_indices, filtered_right_indices)
    """
    # align radec to indices
    left_radec = left[[left_catalog_info.ra_column, left_catalog_info.dec_column]]
    left_radec_aligned = left_radec.iloc[left_idx]
    right_radec = right[[right_catalog_info.ra_column, right_catalog_info.dec_column]]
    right_radec_aligned = right_radec.iloc[right_idx]

    # store the indices from each row
    distances_df = pd.DataFrame.from_dict({"_left_idx": left_idx, "_right_idx": right_idx})

    # calculate distances of each pair
    distances_df["_DIST"] = gc_dist(
        left_radec_aligned[left_catalog_info.ra_column].values,
        left_radec_aligned[left_catalog_info.dec_column].values,
        right_radec_aligned[right_catalog_info.ra_column].values,
        right_radec_aligned[right_catalog_info.dec_column].values,
    )
    # cull based on the distance threshold
    distances_df = distances_df.loc[distances_df["_DIST"] < d_thresh]
    left_ids_filtered = distances_df["_left_idx"]
    right_ids_filtered = distances_df["_right_idx"]
    distances = distances_df["_DIST"].to_numpy()
    return distances, left_ids_filtered, right_ids_filtered


def gc_dist(lon1, lat1, lon2, lat2):
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


def frame_gnomonic(df, catalog_info, clon, clat):
    """
    method taken from lsd1:
    creates a np.array of gnomonic distances for each source in the dataframe
    from the center of the ordered pixel. These values are passed into
    the kdtree NN query during the xmach routine.
    """
    phi = np.radians(df[catalog_info.dec_column].values)
    l = np.radians(df[catalog_info.ra_column].values)
    phi1 = np.radians(clat)
    l0 = np.radians(clon)

    cosc = np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(l - l0)
    x = np.cos(phi) * np.sin(l - l0) / cosc
    y = (
        np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(l - l0)
    ) / cosc

    ret = np.column_stack((np.degrees(x), np.degrees(y)))
    del phi, l, phi1, l0, cosc, x, y
    return ret
