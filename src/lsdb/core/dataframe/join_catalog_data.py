from __future__ import annotations

from typing import TYPE_CHECKING

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
    partitions = pixels.apply(lambda row: catalog.get_partition(row[order_col], row[pixel_col]), axis=1)
    partitions_list = partitions.to_list()
    return dd.concat(partitions_list)


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
    return dd.map_partitions(perform_join, left_aligned_to_join_partitions, right_aligned_to_join_partitions, association_aligned_to_join_partitions, align_dataframes=False, transform_divisions=False)
