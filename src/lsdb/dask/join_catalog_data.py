# pylint: disable=duplicate-code

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Tuple

import dask
import dask.dataframe as dd
import hipscat as hc
import pandas as pd
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN
from hipscat.pixel_tree import PixelAlignment

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_hipscat_index_to_pixel,
    generate_meta_df_for_joined_tables,
    get_healpix_pixels_from_alignment,
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


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_join_on(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_margin: pd.DataFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    left_structure: hc.catalog.Catalog,
    right_structure: hc.catalog.Catalog,
    right_margin_structure: hc.catalog.Catalog,
    left_on: str,
    right_on: str,
    suffixes: Tuple[str, str],
    right_columns: List[str],
):
    """Performs a join on two catalog partitions

    Args:
        left (pd.DataFrame): the left partition to merge
        right (pd.DataFrame): the right partition to merge
        right_margin (pd.DataFrame): the right margin partition to merge
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        right_margin_pixel (HealpixPixel): the HEALPix pixel of the right margin partition
        left_structure (hc.Catalog): the hipscat structure of the left catalog
        right_structure (hc.Catalog): the hipscat structure of the right catalog
        right_margin_structure (hc.Catalog): the hipscat structure of the right margin catalog
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        right_columns (List[str]): the columns to include from the right margin partition

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_hipscat_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    right_joined_df = concat_partition_and_margin(right, right_margin, right_columns)

    left, right_joined_df = rename_columns_with_suffixes(left, right_joined_df, suffixes)
    merged = left.reset_index().merge(
        right_joined_df, left_on=left_on + suffixes[0], right_on=right_on + suffixes[1]
    )
    merged.set_index(HIPSCAT_ID_COLUMN, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_join_through(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_margin: pd.DataFrame,
    through: pd.DataFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    through_pixel: HealpixPixel,
    left_catalog: hc.catalog.Catalog,
    right_catalog: hc.catalog.Catalog,
    right_margin_catalog: hc.catalog.Catalog,
    assoc_catalog: hc.catalog.AssociationCatalog,
    suffixes: Tuple[str, str],
    right_columns: List[str],
):
    """Performs a join on two catalog partitions through an association catalog

    Args:
        left (pd.DataFrame): the left partition to merge
        right (pd.DataFrame): the right partition to merge
        right_margin (pd.DataFrame): the right margin partition to merge
        through (pd.DataFrame): the association column partition to merge with
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        right_margin_pixel (HealpixPixel): the HEALPix pixel of the right margin partition
        through_pixel (HealpixPixel): the HEALPix pixel of the association partition
        left_catalog (hc.Catalog): the hipscat structure of the left catalog
        right_catalog (hc.Catalog): the hipscat structure of the right catalog
        right_margin_catalog (hc.Catalog): the hipscat structure of the right margin catalog
        assoc_catalog (hc.AssociationCatalog): the hipscat structure of the association catalog
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        right_columns (List[str]): the columns to include from the right margin partition

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    catalog_info = assoc_catalog.catalog_info
    if catalog_info.primary_column is None or catalog_info.join_column is None:
        raise ValueError("Invalid catalog_info")
    if right_pixel.order > left_pixel.order:
        left = filter_by_hipscat_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    right_joined_df = concat_partition_and_margin(right, right_margin, right_columns)

    left, right_joined_df = rename_columns_with_suffixes(left, right_joined_df, suffixes)

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
            right_joined_df,
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
        left (lsdb.Catalog): the left catalog to join
        right (lsdb.Catalog): the right catalog to join
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names

    Returns:
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    if right.margin is None:
        warnings.warn("Right catalog does not have a margin cache. Results may be inaccurate", RuntimeWarning)

    alignment = align_catalogs(left, right)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels)],
        perform_join_on,
        left_on,
        right_on,
        suffixes,
        right.columns,
    )

    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes)

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def join_catalog_data_through(
    left: Catalog, right: Catalog, association: AssociationCatalog, suffixes: Tuple[str, str]
) -> Tuple[dd.core.DataFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs with an association table

    Args:
        left (lsdb.Catalog): the left catalog to join
        right (lsdb.Catalog): the right catalog to join
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

    if right.margin is None:
        warnings.warn("Right catalog does not have a margin cache. Results may be inaccurate", RuntimeWarning)

    alignment = align_catalogs(left, right)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [
            (left, left_pixels),
            (right, right_pixels),
            (right.margin, right_pixels),
            (association, left_pixels),
        ],
        perform_join_through,
        suffixes,
        right.columns,
    )

    association_join_columns = [
        association.hc_structure.catalog_info.primary_column_association,
        association.hc_structure.catalog_info.join_column_association,
    ]
    # pylint: disable=protected-access
    extra_df = association._ddf._meta.drop(NON_JOINING_ASSOCIATION_COLUMNS + association_join_columns, axis=1)
    meta_df = generate_meta_df_for_joined_tables([left, extra_df, right], [suffixes[0], "", suffixes[1]])

    return construct_catalog_args(joined_partitions, meta_df, alignment)
