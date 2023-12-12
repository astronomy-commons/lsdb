# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import dask
import dask.dataframe as dd
import pandas as pd
from hipscat.catalog.association_catalog import PartitionJoinInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.dask.divisions import get_pixels_divisions
from lsdb.catalog.association_catalog import AssociationCatalog
import hipscat as hc

from lsdb.dask.join_catalogs import get_healpix_pixels_from_alignment, \
    get_partition_map_from_alignment_pixels, generate_meta_df_for_joined_tables, align_catalog_to_partitions, \
    filter_by_hipscat_index_to_pixel

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog, DaskDFPixelMap


@dask.delayed
def perform_join_on(
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_on: str,
        right_on: str,
        left_pixel: HealpixPixel,
        right_pixel: HealpixPixel,
        suffixes: Tuple[str, str]
):
    """Performs a join on two catalog partitions

    Args:
        left (pd.DataFrame): the left partition to merge
        right (pd.DataFrame): the right partition to merge
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_hipscat_index_to_pixel(left, right_pixel.order, right_pixel.pixel)
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)
    merged = left.reset_index().merge(right, left_on=left_on + suffixes[0], right_on=right_on + suffixes[1])
    merged.set_index(HIPSCAT_ID_COLUMN, inplace=True)
    return merged


@dask.delayed
def perform_join_through(
        left: pd.DataFrame,
        right: pd.DataFrame,
        through: pd.DataFrame,
        left_pixel: HealpixPixel,
        right_pixel: HealpixPixel,
        catalog_info: hc.catalog.association_catalog.AssociationCatalogInfo,
        suffixes: Tuple[str, str]
):
    """Performs a join on two catalog partitions

    Args:
        left (pd.DataFrame): the left partition to merge
        right (pd.DataFrame): the right partition to merge
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_hipscat_index_to_pixel(left, right_pixel.order, right_pixel.pixel)
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)

    join_columns = [catalog_info.primary_column_association]
    if catalog_info.join_column_association != catalog_info.primary_column_association:
        join_columns.append(catalog_info.join_column_association)

    through = through.drop(["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"], axis=1)

    merged = left \
        .reset_index() \
        .merge(through, left_on=catalog_info.primary_column + suffixes[0],
               right_on=catalog_info.primary_column_association) \
        .merge(right, left_on=catalog_info.join_column_association,
               right_on=catalog_info.join_column + suffixes[1])

    merged.set_index(HIPSCAT_ID_COLUMN, inplace=True)
    merged.drop(join_columns, axis=1, inplace=True)
    return merged


def join_catalog_data_on(
        left: Catalog,
        right: Catalog,
        left_on: str,
        right_on: str,
        suffixes: Tuple[str, str]
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs spatially on a specified column

    Args:
        left (Catalog): the left catalog to join
        right (Catalog): the right catalog to join
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    alignment = align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER
    )
    join_pixels = alignment.pixel_mapping
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

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(join_pixels)

    joined_partitions = [
        perform_join_on(left_df, right_df, left_on, right_on, left_pixel, right_pixel, suffixes)
        for left_df, right_df, left_pixel, right_pixel
        in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, left_pixels, right_pixels)
    ]

    partition_map = get_partition_map_from_alignment_pixels(join_pixels)
    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes)
    divisions = get_pixels_divisions(list(partition_map.keys()))
    ddf = dd.from_delayed(joined_partitions, meta=meta_df, divisions=divisions)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment


def get_alignment_from_association(association: AssociationCatalog) -> PixelAlignment:
    smallest_pixels = association.hc_structure.join_info.data_frame.apply(
        lambda row: [row[PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME], row[PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME]] if row[PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME] > row[PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME] else [row[PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME], row[PartitionJoinInfo.JOIN_PIXEL_COLUMN_NAME]],
        axis=1, result_type="expand"
    )
    print(smallest_pixels)
    pixel_mapping = association.hc_structure.join_info.data_frame.assign(**{PixelAlignment.ALIGNED_ORDER_COLUMN_NAME: smallest_pixels[0].values, PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME: smallest_pixels[1].values})
    pixel_mapping.rename(columns={
        PartitionJoinInfo.PRIMARY_ORDER_COLUMN_NAME: PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
        PartitionJoinInfo.PRIMARY_PIXEL_COLUMN_NAME: PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
        PartitionJoinInfo.JOIN_ORDER_COLUMN_NAME: PixelAlignment.JOIN_ORDER_COLUMN_NAME,
        PartitionJoinInfo.JOIN_PIXEL_COLUMN_NAME: PixelAlignment.JOIN_PIXEL_COLUMN_NAME,
    }, inplace=True)
    pixels = [
        HealpixPixel(row[PixelAlignment.ALIGNED_ORDER_COLUMN_NAME], row[PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME])
        for _, row in pixel_mapping.iterrows()
    ]
    tree = PixelTreeBuilder.from_healpix(pixels)
    alignment = PixelAlignment(tree, pixel_mapping, PixelAlignmentType.INNER)
    return alignment


def join_catalog_data_through(
        left: Catalog,
        right: Catalog,
        association: AssociationCatalog,
        suffixes: Tuple[str, str]
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    if not association.hc_structure.catalog_info.contains_leaf_files:
        return join_catalog_data_on(
            left,
            right,
            association.hc_structure.catalog_info.primary_column,
            association.hc_structure.catalog_info.join_column,
            suffixes
        )
    # alignment = get_alignment_from_association(association)
    alignment = align_trees(
        left.hc_structure.pixel_tree,
        right.hc_structure.pixel_tree,
        alignment_type=PixelAlignmentType.INNER
    )
    join_pixels = alignment.pixel_mapping
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
    association_aligned_to_join_partitions = align_catalog_to_partitions(
        association,
        join_pixels,
        order_col=PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    )

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(join_pixels)

    joined_partitions = [
        perform_join_through(left_df, right_df, association_df, left_pixel, right_pixel,
                             association.hc_structure.catalog_info, suffixes)
        for left_df, right_df, association_df, left_pixel, right_pixel
        in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions,
               association_aligned_to_join_partitions, left_pixels, right_pixels)
    ]
    partition_map = get_partition_map_from_alignment_pixels(alignment.pixel_mapping)
    extra_cols = {}
    for name, col_type in association._ddf.dtypes.items():
        if name not in ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix", association.hc_structure.catalog_info.primary_column_association, association.hc_structure.catalog_info.join_column_association]:
            extra_cols[name] = pd.Series(dtype=col_type)
    extra_df = pd.DataFrame(extra_cols)
    meta_df = generate_meta_df_for_joined_tables([left, extra_df, right], [suffixes[0], "", suffixes[1]])
    ddf = dd.from_delayed(joined_partitions, meta=meta_df)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment
