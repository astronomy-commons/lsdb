from __future__ import annotations

from typing import TYPE_CHECKING

import dask
import dask.dataframe as dd
import pandas as pd
from hipscat.catalog.association_catalog.partition_join_info import \
    PartitionJoinInfo

if TYPE_CHECKING:
    from lsdb.catalog.association_catalog.association_catalog import \
        AssociationCatalog
    from lsdb.catalog.catalog import Catalog


def align_catalog_to_partitions(
        catalog: Catalog, pixels: pd.DataFrame, order_col: str = "Norder", pixel_col: str = "Npix"
) -> dd.core.DataFrame:
    dfs = catalog._ddf.to_delayed()
    partitions = pixels.apply(lambda row: dfs[catalog.get_partition_index(row[order_col], row[pixel_col])], axis=1)
    partitions_list = partitions.to_list()
    return partitions_list


@dask.delayed
def perform_join(left: pd.DataFrame, right: pd.DataFrame, through: pd.DataFrame):
    return left.merge(through, left_index=True, right_index=True).merge(right, left_on="join_hipscat_id", right_index=True)


def join_catalog_data(
        left: Catalog, right: Catalog, through: AssociationCatalog
) -> dd.core.DataFrame:
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
        pixel_col=PartitionJoinInfo.JOIN_PIXEL_PIXEL_NAME,
    )
    association_aligned_to_join_partitions = align_catalog_to_partitions(
        through,
        join_pixels,
        order_col=PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME,
    )
    joined_partitions = [perform_join(left_df, right_df, join_df) for left_df, right_df, join_df in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, association_aligned_to_join_partitions)]
    return dd.from_delayed(joined_partitions)
