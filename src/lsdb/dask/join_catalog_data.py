from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import dask
import dask.dataframe as dd
import pandas as pd
import hipscat as hc
from hipscat.catalog.association_catalog.partition_join_info import \
    PartitionJoinInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_tree import PixelAlignmentType, PixelAlignment

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
def perform_join_on(left: pd.DataFrame, right: pd.DataFrame, on: str):
    return left.merge(right, on=on, suffixes=("left", "right"))


@dask.delayed
def filter_index_to_range(df: pd.DataFrame, lower: int, upper: int):
    return df.loc[lower:upper]


@dask.delayed
def concat_dfs(dfs: List[pd.DataFrame]):
    return pd.concat(dfs).sort_index()


def join_catalog_data(
        left: Catalog, right: Catalog, through: AssociationCatalog, suffixes: Tuple[str, str] | None = None
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    if suffixes is None:
        suffixes = ("", "")
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
    meta = {}
    for name, t in left._ddf.dtypes.items():
        meta[name + suffixes[0]] = pd.Series(dtype=t)
    for name, t in through._ddf.dtypes.items():
        meta[name] = pd.Series(dtype=t)
    for name, t in right._ddf.dtypes.items():
        meta[name + suffixes[1]] = pd.Series(dtype=t)
    meta_df = pd.DataFrame(meta)
    meta_df.index.name = "_hipscat_index"
    ddf = dd.from_delayed(final_partitions, meta=meta_df)
    return ddf, partition_map, alignment
