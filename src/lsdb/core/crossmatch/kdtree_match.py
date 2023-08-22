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
        n_neighbors=1,
        dthresh=0.01):

    left = left.copy(deep=False)
    right = right.copy(deep=False)

    (clon, clat) = hp.pix2ang(hp.order2nside(left_order), left_pixel, nest=True, lonlat=True)
    left_md = {'ra_kw': left_metadata.catalog_info.ra_column,
               'dec_kw': left_metadata.catalog_info.dec_column}
    right_md = {'ra_kw': right_metadata.catalog_info.ra_column,
                'dec_kw': right_metadata.catalog_info.dec_column}
    xy1 = frame_gnomonic(left, left_md, clon, clat)
    xy2 = frame_gnomonic(right, right_md, clon, clat)

    # construct the KDTree from the comparative catalog: c2/xy2
    tree = KDTree(xy2, leaf_size=2)
    # find the indicies for the nearest neighbors
    # this is the cross-match calculation
    dists, inds = tree.query(xy1, k=min([n_neighbors, len(xy2)]))

    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left.rename(columns=left_columns_renamed, inplace=True)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right.rename(columns=right_columns_renamed, inplace=True)

    # numpy indice magic for the joining of the two catalogs
    outIdx = np.arange(len(left) * n_neighbors)  # index of each row in the output table (0... number of output rows)
    leftIdx = outIdx // n_neighbors  # index of the corresponding row in the left table (0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
    rightIdx = inds.ravel()  # index of the corresponding row in the right table (22, 33, 44, 55, 66, ...)
    left.index.name = "_hipscat_index"
    left_radec = left[[left_md['ra_kw'] + suffixes[0], left_md['dec_kw'] + suffixes[0]]]
    left_radec_join_part = left_radec.iloc[leftIdx].reset_index(drop=True)
    right_radec = right[[right_md['ra_kw'] + suffixes[1], right_md['dec_kw'] + suffixes[1]]]
    right_radec_join_part = right_radec.iloc[rightIdx].reset_index(drop=True)
    distances = pd.concat(
        [
            left_radec_join_part,  # select the rows of the left table
            right_radec_join_part  # select the rows of the right table
        ], axis=1)  # concat the two tables "horizontally" (i.e., join columns, not append rows)
    distances["_left_idx"] = leftIdx
    distances["_right_idx"] = rightIdx

    # save the order/pix/and distances for each nearest neighbor
    distances["_DIST"] = gc_dist(
        distances[left_md['ra_kw'] + suffixes[0]], distances[left_md['dec_kw'] + suffixes[0]],
        distances[right_md['ra_kw'] + suffixes[1]], distances[right_md['dec_kw'] + suffixes[1]]
    )

    # cull the return dataframe based on the distance threshold
    distances = distances.loc[distances['_DIST'] < dthresh]
    left_ids_filtered = distances["_left_idx"]
    right_ids_filtered = distances["_right_idx"]
    left_join_part = left.iloc[left_ids_filtered].reset_index()
    right_join_part = right.iloc[right_ids_filtered].reset_index(drop=True)
    out = pd.concat(
        [
            left_join_part,  # select the rows of the left table
            right_join_part  # select the rows of the right table
        ], axis=1)
    out.set_index("_hipscat_index", inplace=True)
    out["_DIST"] = distances["_DIST"].to_numpy()

    return out


def gc_dist(lon1, lat1, lon2, lat2):
    '''
        function that calculates the distance between two points
            p1 (lon1, lat1) or (ra1, dec1)
            p2 (lon2, lat2) or (ra2, dec2)

            can be np.array()
            returns np.array()
    '''
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)

    return np.degrees(2*np.arcsin(np.sqrt( (np.sin((lat1-lat2)*0.5))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon1-lon2)*0.5))**2 )))


def frame_gnomonic(df, df_md, clon, clat):
    '''
        method taken from lsd1:
        creates a np.array of gnomonic distances for each source in the dataframe
        from the center of the ordered pixel. These values are passed into
        the kdtree NN query during the xmach routine.
    '''
    phi  = np.radians(df[df_md['dec_kw']].values)
    l    = np.radians(df[df_md['ra_kw']].values)
    phi1 = np.radians(clat)
    l0   = np.radians(clon)

    cosc = np.sin(phi1)*np.sin(phi) + np.cos(phi1)*np.cos(phi)*np.cos(l-l0)
    x = np.cos(phi)*np.sin(l-l0) / cosc
    y = (np.cos(phi1)*np.sin(phi) - np.sin(phi1)*np.cos(phi)*np.cos(l-l0)) / cosc

    ret = np.column_stack((np.degrees(x), np.degrees(y)))
    del phi, l, phi1, l0, cosc, x, y
    return ret