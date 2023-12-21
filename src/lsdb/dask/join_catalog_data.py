# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import dask
import dask.dataframe as dd
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.dask.merge_catalog_functions import (
    align_catalog_to_partitions,
    align_catalogs_to_alignment_mapping,
    filter_by_hipscat_index_to_pixel,
    generate_meta_df_for_joined_tables,
    get_healpix_pixels_from_alignment,
    get_partition_map_from_alignment_pixels,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


NON_JOINING_ASSOCIATION_COLUMNS = ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"]


def rename_columns_with_suffixes(left: pd.DataFrame, right: pd.DataFrame, suffixes: Tuple[str, str]):
    """Renames two dataframes with the suffixes specified

    Args:
        left (pd.DataFrame): the left dataframe to apply the first suffix to
        right (pd.DataFrame): the right dataframe to apply the second suffix to
        suffixes (Tuple[str, str]): the pair of suffixes to apply to the dataframes

    Returns:
        A tuple of (left, right) updated dataframes with their columns renamed
    """
    left_columns_renamed = {name: name + suffixes[0] for name in left.columns}
    left = left.rename(columns=left_columns_renamed)
    right_columns_renamed = {name: name + suffixes[1] for name in right.columns}
    right = right.rename(columns=right_columns_renamed)
    return left, right


@dask.delayed
def perform_join_on(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    suffixes: Tuple[str, str],
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
    left, right = rename_columns_with_suffixes(left, right, suffixes)
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
    suffixes: Tuple[str, str],
):
    """Performs a join on two catalog partitions through an association catalog

    Args:
        left (pd.DataFrame): the left partition to merge
        right (pd.DataFrame): the right partition to merge
        through (pd.DataFrame): the association column partition to merge with
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        catalog_info (AssociationCatalogInfo): the catalog_info of the association catalog
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if catalog_info.primary_column is None or catalog_info.join_column is None:
        raise ValueError("Invalid catalog_info")
    if right_pixel.order > left_pixel.order:
        left = filter_by_hipscat_index_to_pixel(left, right_pixel.order, right_pixel.pixel)
    left, right = rename_columns_with_suffixes(left, right, suffixes)

    join_columns = [catalog_info.primary_column_association]
    if catalog_info.join_column_association != catalog_info.primary_column_association:
        join_columns.append(catalog_info.join_column_association)

    through = through.drop(NON_JOINING_ASSOCIATION_COLUMNS, axis=1)

    merged = (
        left.reset_index()
        .merge(
            through,
            left_on=catalog_info.primary_column + suffixes[0],
            right_on=catalog_info.primary_column_association,
        )
        .merge(
            right,
            left_on=catalog_info.join_column_association,
            right_on=catalog_info.join_column + suffixes[1],
        )
    )

    merged.set_index(HIPSCAT_ID_COLUMN, inplace=True)
    merged.drop(join_columns, axis=1, inplace=True)
    return merged


def join_catalog_data_on(
    left: Catalog, right: Catalog, left_on: str, right_on: str, suffixes: Tuple[str, str]
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
        left.hc_structure.pixel_tree, right.hc_structure.pixel_tree, alignment_type=PixelAlignmentType.INNER
    )
    join_pixels = alignment.pixel_mapping
    left_aligned_partitions, right_aligned_partitions = align_catalogs_to_alignment_mapping(
        join_pixels, left, right
    )

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(join_pixels)

    apply_join = np.vectorize(
        lambda left_df, right_df, left_pix, right_pix: perform_join_on(
            left_df,
            right_df,
            left_on,
            right_on,
            left_pix,
            right_pix,
            suffixes,
        )
    )

    joined_partitions = apply_join(
        left_aligned_partitions, right_aligned_partitions, left_pixels, right_pixels
    )

    partition_map = get_partition_map_from_alignment_pixels(join_pixels)
    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes)
    divisions = get_pixels_divisions(list(partition_map.keys()))
    ddf = dd.from_delayed(joined_partitions, meta=meta_df, divisions=divisions)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment


def join_catalog_data_through(
    left: Catalog, right: Catalog, association: AssociationCatalog, suffixes: Tuple[str, str]
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs with an association table

    Args:
        left (Catalog): the left catalog to join
        right (Catalog): the right catalog to join
        association (AssociationCatalog): the association catalog to join the catalogs with
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.

    """
    if (
        association.hc_structure.catalog_info.primary_column is None
        or association.hc_structure.catalog_info.join_column is None
    ):
        raise ValueError("Invalid catalog_info")

    if not association.hc_structure.catalog_info.contains_leaf_files:
        return join_catalog_data_on(
            left,
            right,
            association.hc_structure.catalog_info.primary_column,
            association.hc_structure.catalog_info.join_column,
            suffixes,
        )
    alignment = align_trees(
        left.hc_structure.pixel_tree, right.hc_structure.pixel_tree, alignment_type=PixelAlignmentType.INNER
    )
    join_pixels = alignment.pixel_mapping
    left_aligned_partitions, right_aligned_partitions = align_catalogs_to_alignment_mapping(
        join_pixels, left, right
    )
    association_aligned_to_join_partitions = align_catalog_to_partitions(
        association,
        join_pixels,
        order_col=PixelAlignment.PRIMARY_ORDER_COLUMN_NAME,
        pixel_col=PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME,
    )

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(join_pixels)

    apply_join = np.vectorize(
        lambda left_df, right_df, assoc_df, left_pix, right_pix: perform_join_through(
            left_df,
            right_df,
            assoc_df,
            left_pix,
            right_pix,
            association.hc_structure.catalog_info,
            suffixes,
        )
    )

    joined_partitions = apply_join(
        left_aligned_partitions,
        right_aligned_partitions,
        association_aligned_to_join_partitions,
        left_pixels,
        right_pixels,
    )

    partition_map = get_partition_map_from_alignment_pixels(alignment.pixel_mapping)
    association_join_columns = [
        association.hc_structure.catalog_info.primary_column_association,
        association.hc_structure.catalog_info.join_column_association,
    ]
    # pylint: disable=protected-access
    extra_df = association._ddf._meta.drop(NON_JOINING_ASSOCIATION_COLUMNS + association_join_columns, axis=1)
    meta_df = generate_meta_df_for_joined_tables([left, extra_df, right], [suffixes[0], "", suffixes[1]])
    divisions = get_pixels_divisions(list(partition_map.keys()))
    ddf = dd.from_delayed(joined_partitions, meta=meta_df, divisions=divisions)
    ddf = cast(dd.DataFrame, ddf)
    return ddf, partition_map, alignment