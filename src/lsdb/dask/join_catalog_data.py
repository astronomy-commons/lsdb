# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.dask.crossmatch_catalog_data import (
    align_catalog_to_partitions,
    filter_by_hipscat_index_to_pixel,
    generate_meta_df_for_joined_tables,
    get_partition_map_from_alignment_pixels,
)
from lsdb.dask.divisions import get_pixels_divisions

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

    left_pixels = [
        HealpixPixel(
            row[PixelAlignment.PRIMARY_ORDER_COLUMN_NAME],
            row[PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME]
        )
        for _, row in join_pixels.iterrows()
    ]

    right_pixels = [
        HealpixPixel(
            row[PixelAlignment.JOIN_ORDER_COLUMN_NAME],
            row[PixelAlignment.JOIN_PIXEL_COLUMN_NAME]
        )
        for _, row in join_pixels.iterrows()
    ]

    joined_partitions = [
        perform_join_on(left_df, right_df, left_on, right_on, left_pixel, right_pixel, suffixes)
        for left_df, right_df, left_pixel, right_pixel
        in zip(left_aligned_to_join_partitions, right_aligned_to_join_partitions, left_pixels, right_pixels)
    ]

    partition_map = get_partition_map_from_alignment_pixels(join_pixels)
    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes)
    pixels = list(partition_map.keys())
    ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
    divisions = get_pixels_divisions(ordered_pixels)
    ddf = dd.from_delayed(joined_partitions, meta=meta_df, divisions=divisions)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment
