from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import hipscat as hc
from hipscat.catalog.association_catalog.partition_join_info import \
    PartitionJoinInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_tree import PixelAlignmentType, PixelAlignment
from sklearn.neighbors import KDTree

import healpy as hp

if TYPE_CHECKING:
    from lsdb.catalog.association_catalog.association_catalog import \
        AssociationCatalog
    from lsdb.catalog.catalog import Catalog, DaskDFPixelMap


def align_catalog_to_partitions(
        catalog: Catalog,
        pixels: pd.DataFrame,
        order_col: str = "Norder",
        pixel_col: str = "Npix"
) -> dd.core.DataFrame:
    dfs = catalog._ddf.to_delayed()
    partitions = pixels.apply(lambda row: dfs[
        catalog.get_partition_index(row[order_col], row[pixel_col])], axis=1)
    partitions_list = partitions.to_list()
    return partitions_list


def align_association_catalog_to_partitions(
        catalog: AssociationCatalog,
        pixels: pd.DataFrame,
        primary_order_col: str = "primary_Norder",
        primary_pixel_col: str = "primary_Npix",
        join_order_col: str = "join_Norder",
        join_pixel_col: str = "join_Npix",
) -> dd.core.DataFrame:
    dfs = catalog._ddf.to_delayed()
    partitions = pixels.apply(
        lambda row: dfs[catalog.get_partition_index((row[primary_order_col], row[primary_pixel_col]), (row[join_order_col], row[join_pixel_col]))]
        , axis=1
    )
    partitions_list = partitions.to_list()
    return partitions_list


@dask.delayed
def perform_join(left: pd.DataFrame, right: pd.DataFrame, through: pd.DataFrame, suffixes: Tuple[str, str]):
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)
    return left.merge(through, left_index=True, right_index=True).merge(right, left_on="join_hipscat_index", right_index=True)


@dask.delayed
def perform_crossmatch(left: pd.DataFrame, right: pd.DataFrame, order: int, pixel: int, suffixes: Tuple[str, str], n_neighbors=1, dthresh=0.01):
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)

    (clon, clat) = hp.pix2ang(hp.order2nside(order), pixel, nest=True, lonlat=True)
    left_md = {'ra_kw': "ra" + suffixes[0], 'dec_kw': "dec" + suffixes[0]}
    right_md = {'ra_kw': "ra" + suffixes[1], 'dec_kw': "dec" + suffixes[1]}
    xy1 = frame_gnomonic(left, left_md, clon, clat)
    xy2 = frame_gnomonic(right, right_md, clon, clat)

    # construct the KDTree from the comparative catalog: c2/xy2
    tree = KDTree(xy2, leaf_size=2)
    # find the indicies for the nearest neighbors
    # this is the cross-match calculation
    dists, inds = tree.query(xy1, k=min([n_neighbors, len(xy2)]))

    # numpy indice magic for the joining of the two catalogs
    outIdx = np.arange(
        len(left) * n_neighbors)  # index of each row in the output table (0... number of output rows)
    leftIdx = outIdx // n_neighbors  # index of the corresponding row in the left table (0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
    rightIdx = inds.ravel()  # index of the corresponding row in the right table (22, 33, 44, 55, 66, ...)
    left.index.name = "_hipscat_index"
    out = pd.concat(
        [
            left.iloc[leftIdx].reset_index(),  # select the rows of the left table
            right.iloc[rightIdx].reset_index(drop=True)  # select the rows of the right table
        ], axis=1)  # concat the two tables "horizontally" (i.e., join columns, not append rows)
    out = out.set_index("_hipscat_index")

    # save the order/pix/and distances for each nearest neighbor
    out["_DIST"] = gc_dist(
        out[left_md['ra_kw']], out[left_md['dec_kw']],
        out[right_md['ra_kw']], out[right_md['dec_kw']]
    )

    # cull the return dataframe based on the distance threshold
    out = out.loc[out['_DIST'] < dthresh]
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


@dask.delayed
def perform_join_on(left: pd.DataFrame, right: pd.DataFrame, on: str):
    return left.merge(right, on=on, suffixes=("left", "right"))


@dask.delayed
def filter_index_to_range(df: pd.DataFrame, lower: int, upper: int):
    return df.loc[lower:upper]


@dask.delayed
def concat_dfs(dfs: List[pd.DataFrame]):
    return pd.concat(dfs).sort_index()


def join_catalog_data(
        left: Catalog, right: Catalog, through: AssociationCatalog, suffixes: Tuple[str, str] = ("_left", "_right")
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    join_pixels = through.hc_structure.get_join_pixels()
    left_aligned_to_join_partitions = align_catalog_to_partitions(
        left,
        join_pixels,
        order_col=PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME,
    )
    right_aligned_to_join_partitions = align_catalog_to_partitions(
        right,
        join_pixels,
        order_col=PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME,
        pixel_col=PartitionJoinInfo.JOIN_PIXEL_COLUMN_NAME,
    )
    association_aligned_to_join_partitions = align_association_catalog_to_partitions(
        through,
        join_pixels,
        primary_order_col=PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME,
        primary_pixel_col=PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME,
        join_order_col=PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME,
        join_pixel_col=PartitionJoinInfo.JOIN_PIXEL_COLUMN_NAME,
    )
    joined_partitions = [perform_join(left_df, right_df, join_df, suffixes) for left_df, right_df, join_df in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, association_aligned_to_join_partitions)]
    alignment = PixelAlignment.align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.LEFT
    )
    indexed_join_pixels = join_pixels.reset_index()
    final_partitions = []
    partition_index = 0
    partition_map = {}
    for _, row in alignment.pixel_mapping.iterrows():
        left_order = row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME]
        left_pixel = row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME]
        aligned_order = int(row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME])
        aligned_pixel = int(row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME])
        left_indexes = indexed_join_pixels.index[
            (indexed_join_pixels[PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME] == left_order)
            & (indexed_join_pixels[PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME] == left_pixel)
            ].tolist()
        partitions_to_filter = [joined_partitions[i] for i in left_indexes]
        lower_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(aligned_order, aligned_pixel)
        upper_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(aligned_order, aligned_pixel+1)
        filtered_partitions = [filter_index_to_range(partition, lower_bound, upper_bound) for partition in partitions_to_filter]
        final_partitions.append(concat_dfs(filtered_partitions))
        final_pixel = HealpixPixel(aligned_order, aligned_pixel)
        partition_map[final_pixel] = partition_index
        partition_index += 1
    meta = []
    for name, t in left._ddf.dtypes.items():
        meta.append((name + suffixes[0], t))
    for name, t in through._ddf.dtypes.items():
        meta.append((name, t))
    for name, t in right._ddf.dtypes.items():
        meta.append((name + suffixes[1], t))
    ddf = dd.from_delayed(final_partitions, meta=meta)
    return ddf, partition_map, alignment


def crossmatch_catalog_data(
        left: Catalog, right: Catalog, suffixes: Tuple[str, str] | None = None
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    if suffixes is None:
        suffixes = ("", "")
    join_pixels = PixelAlignment.align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER
    ).pixel_mapping
    left_aligned_to_join_partitions = align_catalog_to_partitions(
        left,
        join_pixels,
        order_col=PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    )
    right_aligned_to_join_partitions = align_catalog_to_partitions(
        right,
        join_pixels,
        order_col=PixelAlignment.JOIN_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
    )
    orders = [row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    pixels = [row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME] for _, row in join_pixels.iterrows()]
    joined_partitions = [perform_crossmatch(left_df, right_df, order, pixel, suffixes) for left_df, right_df, order, pixel in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, orders, pixels)]
    alignment = PixelAlignment.align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.LEFT
    )
    indexed_join_pixels = join_pixels.reset_index()
    final_partitions = []
    partition_index = 0
    partition_map = {}
    for _, row in alignment.pixel_mapping.iterrows():
        left_order = row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME]
        left_pixel = row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME]
        aligned_order = int(row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME])
        aligned_pixel = int(row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME])
        left_indexes = indexed_join_pixels.index[
            (indexed_join_pixels[PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME] == left_order)
            & (indexed_join_pixels[PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME] == left_pixel)
            ].tolist()
        partitions_to_filter = [joined_partitions[i] for i in left_indexes]
        lower_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(aligned_order, aligned_pixel)
        upper_bound = hc.pixel_math.hipscat_id.healpix_to_hipscat_id(aligned_order, aligned_pixel+1)
        filtered_partitions = [filter_index_to_range(partition, lower_bound, upper_bound) for partition in partitions_to_filter]
        final_partitions.append(concat_dfs(filtered_partitions))
        final_pixel = HealpixPixel(aligned_order, aligned_pixel)
        partition_map[final_pixel] = partition_index
        partition_index += 1
    meta = []
    for name, t in left._ddf.dtypes.items():
        meta.append((name + suffixes[0], t))
    for name, t in right._ddf.dtypes.items():
        meta.append((name + suffixes[1], t))
    meta.append(('_DIST', np.dtype("float64")))
    ddf = dd.from_delayed(final_partitions, meta=meta)
    return ddf, partition_map, alignment


def partition_joined_data_to_structure(dataframe: dd.DataFrame, df_pixels: pd.DataFrame, joined_structure: hc.catalog.Catalog) -> tuple[dd.DataFrame, DaskDFPixelMap]:
    pass


def join_catalogs(
        left: Catalog, right: Catalog, through: AssociationCatalog
) -> Catalog:
    joined_raw_dataframe = join_catalog_data(left, right, through)
    catalog_alignment = hc.catalog.align_catalogs(left.hc_structure, right.hc_structure)
    joined_structure = hc.Catalog(catalog_alignment.get_pixel_tree())
    joined_aligned_dataframe, joined_df_pixel_map = partition_joined_data_to_structure(joined_raw_dataframe, through.hc_structure.get_join_pixels(), joined_structure)
    return Catalog(joined_aligned_dataframe, joined_df_pixel_map, joined_structure)
