# pylint: disable=duplicate-code
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Tuple

import dask
import nested_dask as nd
import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from hats.pixel_tree import PixelAlignment
from nested_pandas.series.packer import pack_flat

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


NON_JOINING_ASSOCIATION_COLUMNS = ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"]


def rename_columns_with_suffixes(left: npd.NestedFrame, right: npd.NestedFrame, suffixes: Tuple[str, str]):
    """Renames two dataframes with the suffixes specified

    Args:
        left (npd.NestedFrame): the left dataframe to apply the first suffix to
        right (npd.NestedFrame): the right dataframe to apply the second suffix to
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
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    right_margin: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    left_on: str,
    right_on: str,
    suffixes: Tuple[str, str],
    right_columns: List[str],
):
    """Performs a join on two catalog partitions

    Args:
        left (npd.NestedFrame): the left partition to merge
        right (npd.NestedFrame): the right partition to merge
        right_margin (npd.NestedFrame): the right margin partition to merge
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        right_margin_pixel (HealpixPixel): the HEALPix pixel of the right margin partition
        left_catalog_info (hc.TableProperties): the catalog info of the left catalog
        right_catalog_info (hc.TableProperties): the catalog info of the right catalog
        right_margin_catalog_info (hc.TableProperties): the catalog info of the right margin catalog
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        right_columns (List[str]): the columns to include from the right margin partition

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    right_joined_df = concat_partition_and_margin(right, right_margin, right_columns)

    left, right_joined_df = rename_columns_with_suffixes(left, right_joined_df, suffixes)
    merged = left.reset_index().merge(
        right_joined_df, left_on=left_on + suffixes[0], right_on=right_on + suffixes[1]
    )
    merged.set_index(SPATIAL_INDEX_COLUMN, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_join_nested(
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    right_margin: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    left_on: str,
    right_on: str,
    right_columns: List[str],
    right_name: str,
):
    """Performs a join on two catalog partitions by adding the right catalog a nested column using
    nested-pandas

    Args:
        left (npd.NestedFrame): the left partition to merge
        right (npd.NestedFrame): the right partition to merge
        right_margin (npd.NestedFrame): the right margin partition to merge
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        right_margin_pixel (HealpixPixel): the HEALPix pixel of the right margin partition
        left_catalog_info (hc.TableProperties): the catalog info of the left catalog
        right_catalog_info (hc.TableProperties): the catalog info of the right catalog
        right_margin_catalog_info (hc.TableProperties): the catalog info of the right margin catalog
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        right_columns (List[str]): the columns to include from the right margin partition
        right_name (str): the name of the nested column in the resulting df to join the right catalog into

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    right_joined_df = concat_partition_and_margin(right, right_margin, right_columns)

    right_joined_df = pack_flat(npd.NestedFrame(right_joined_df.set_index(right_on))).rename(right_name)

    merged = left.reset_index().merge(right_joined_df, left_on=left_on, right_index=True)
    merged.set_index(SPATIAL_INDEX_COLUMN, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_join_through(
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    right_margin: npd.NestedFrame,
    through: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    through_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    assoc_catalog_info: TableProperties,
    suffixes: Tuple[str, str],
    right_columns: List[str],
):
    """Performs a join on two catalog partitions through an association catalog

    Args:
        left (npd.NestedFrame): the left partition to merge
        right (npd.NestedFrame): the right partition to merge
        right_margin (npd.NestedFrame): the right margin partition to merge
        through (npd.NestedFrame): the association column partition to merge with
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        right_margin_pixel (HealpixPixel): the HEALPix pixel of the right margin partition
        through_pixel (HealpixPixel): the HEALPix pixel of the association partition
        left_catalog_info (hc.TableProperties): the hats structure of the left catalog
        right_catalog_info (hc.TableProperties): the hats structure of the right catalog
        right_margin_catalog_info (hc.TableProperties): the hats structure of the right margin
            catalog
        assoc_catalog_info (hc.TableProperties): the hats structure of the association catalog
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        right_columns (List[str]): the columns to include from the right margin partition

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if assoc_catalog_info.primary_column is None or assoc_catalog_info.join_column is None:
        raise ValueError("Invalid catalog_info")
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    right_joined_df = concat_partition_and_margin(right, right_margin, right_columns)

    left, right_joined_df = rename_columns_with_suffixes(left, right_joined_df, suffixes)

    join_columns = [assoc_catalog_info.primary_column_association]
    if assoc_catalog_info.join_column_association != assoc_catalog_info.primary_column_association:
        join_columns.append(assoc_catalog_info.join_column_association)

    through = through.drop(NON_JOINING_ASSOCIATION_COLUMNS, axis=1)

    merged = (
        left.reset_index()
        .merge(
            through,
            left_on=assoc_catalog_info.primary_column + suffixes[0],
            right_on=assoc_catalog_info.primary_column_association,
        )
        .merge(
            right_joined_df,
            left_on=assoc_catalog_info.join_column_association,
            right_on=assoc_catalog_info.join_column + suffixes[1],
        )
    )

    merged.set_index(SPATIAL_INDEX_COLUMN, inplace=True)
    merged.drop(join_columns, axis=1, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_merge_asof(
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    suffixes: Tuple[str, str],
    direction: str,
):
    """Performs a merge_asof on two catalog partitions

    Args:
        left (npd.NestedFrame): the left partition to merge
        right (npd.NestedFrame): the right partition to merge
        left_pixel (HealpixPixel): the HEALPix pixel of the left partition
        right_pixel (HealpixPixel): the HEALPix pixel of the right partition
        left_catalog_info (hc.TableProperties): the catalog info of the left catalog
        right_catalog_info (hc.TableProperties): the catalog info of the right catalog
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        direction (str): The direction to perform the merge_asof

    Returns:
        A dataframe with the result of merging the left and right partitions on the specified columns with
        `merge_asof`
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(left, right_pixel.order, right_pixel.pixel)

    left, right = rename_columns_with_suffixes(left, right, suffixes)
    left.sort_index(inplace=True)
    right.sort_index(inplace=True)
    merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction=direction)
    return merged


def join_catalog_data_on(
    left: Catalog, right: Catalog, left_on: str, right_on: str, suffixes: Tuple[str, str]
) -> Tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
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
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

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


def join_catalog_data_nested(
    left: Catalog,
    right: Catalog,
    left_on: str,
    right_on: str,
    nested_column_name: str | None = None,
) -> Tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs spatially on a specified column, adding the right as a nested column with nested
    dask

    Args:
        left (lsdb.Catalog): the left catalog to join
        right (lsdb.Catalog): the right catalog to join
        left_on (str): the column to join on from the left partition
        right_on (str): the column to join on from the right partition
        nested_column_name (str): the name of the nested column in the final output, if None, defaults to
            name of the right catalog

    Returns:
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    if nested_column_name is None:
        nested_column_name = right.name

    alignment = align_catalogs(left, right)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels)],
        perform_join_nested,
        left_on,
        right_on,
        right.columns,
        nested_column_name,
    )

    meta_df = generate_meta_df_for_nested_tables([left], right, nested_column_name, join_column_name=right_on)

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def join_catalog_data_through(
    left: Catalog, right: Catalog, association: AssociationCatalog, suffixes: Tuple[str, str]
) -> Tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
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
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

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


def merge_asof_catalog_data(
    left: Catalog, right: Catalog, suffixes: Tuple[str, str], direction: str = "backward"
) -> Tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Uses the pandas `merge_asof` function to merge two catalogs on their indices by distance of keys

    Must be along catalog indices, and does not include margin caches, meaning results may be incomplete for
    merging points.

    This function is intended for use in special cases such as Dust Map Catalogs, for general merges,
    the `crossmatch`and `join` functions should be used.

    Args:
        left (lsdb.Catalog): the left catalog to join
        right (lsdb.Catalog): the right catalog to join
        suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
        direction (str): the direction to perform the merge_asof

    Returns:
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """

    alignment = align_catalogs(left, right)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels)], perform_merge_asof, suffixes, direction
    )

    meta_df = generate_meta_df_for_joined_tables([left, right], suffixes)

    return construct_catalog_args(joined_partitions, meta_df, alignment)
