# pylint: disable=duplicate-code
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment, PixelAlignmentType
from nested_pandas.series.packer import pack_flat

import lsdb.nested as nd
from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    align_catalogs_with_association,
    apply_left_suffix,
    apply_right_suffix,
    apply_suffixes,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_aligned_pixels_from_alignment,
    get_healpix_pixels_from_alignment,
    get_healpix_pixels_from_association,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog

NON_JOINING_ASSOCIATION_COLUMNS = ["Norder", "Dir", "Npix", "join_Norder", "join_Dir", "join_Npix"]


# pylint: disable=too-many-arguments, unused-argument
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
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
):
    """Performs a join on two catalog partitions

    Parameters
    ----------
    left : npd.NestedFrame
        The left partition to merge
    right : npd.NestedFrame
        The right partition to merge
    right_margin : npd.NestedFrame
        The right margin partition to merge
    left_pixel : HealpixPixel
        The HEALPix pixel of the left partition
    right_pixel : HealpixPixel
        The HEALPix pixel of the right partition
    right_margin_pixel : HealpixPixel
        The HEALPix pixel of the right margin partition
    left_catalog_info : hc.TableProperties
        The catalog info of the left catalog
    right_catalog_info : hc.TableProperties
        The catalog info of the right catalog
    right_margin_catalog_info : hc.TableProperties
        The catalog info of the right margin catalog
    left_on : str
        The column to join on from the left partition
    right_on : str
        The column to join on from the right partition
    suffixes : tuple[str, str]
        The suffixes to apply to each partition's column names
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

    Returns
    -------
    npd.NestedFrame
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(
            left, right_pixel.order, right_pixel.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    right_joined_df = concat_partition_and_margin(right, right_margin)

    left_join_column = apply_left_suffix(left_on, right_joined_df.columns, suffixes, suffix_method)
    right_join_column = apply_right_suffix(right_on, left.columns, suffixes, suffix_method)
    left, right_joined_df = apply_suffixes(left, right_joined_df, suffixes, suffix_method, log_changes=False)

    merged = left.reset_index().merge(right_joined_df, left_on=left_join_column, right_on=right_join_column)
    merged.set_index(left_catalog_info.healpix_column, inplace=True)
    return merged


# pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument
def perform_join_nested(
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    right_margin: npd.NestedFrame,
    aligned_df: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    right_margin_pixel: HealpixPixel,
    aligned_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    aligned_catalog_info: TableProperties | None,
    left_on: str,
    right_on: str,
    right_name: str,
    right_meta: npd.NestedFrame,
    how: str,
):
    """Performs a join on two catalog partitions by adding the right catalog a nested column using
    nested-pandas

    Parameters
    ----------
    left : npd.NestedFrame
        The left partition to merge
    right : npd.NestedFrame
        The right partition to merge
    right_margin : npd.NestedFrame
        The right margin partition to merge
    aligned_df : npd.NestedFrame
        The partition of the aligned pixel
    left_pixel : HealpixPixel
        The HEALPix pixel of the left partition
    right_pixel : HealpixPixel
        The HEALPix pixel of the right partition
    right_margin_pixel : HealpixPixel
        The HEALPix pixel of the right margin partition
    aligned_pixel : HealpixPixel
        The HEALPix pixel of the aligned partition
    left_catalog_info : hc.TableProperties
        The catalog info of the left catalog
    right_catalog_info : hc.TableProperties
        The catalog info of the right catalog
    right_margin_catalog_info : hc.TableProperties
        The catalog info of the right margin catalog
    aligned_catalog_info : hc.TableProperties | None
        The catalog info of the aligned catalog; usually None
    left_on : str
        The column to join on from the left partition
    right_on : str
        The column to join on from the right partition
    right_name : str
        The name of the nested column in the resulting df to join the right catalog into
    right_meta : npd.NestedFrame
        The meta for the right catalog (needed for how=`left`)
    how : One of {'inner', 'left'}, default 'inner'
        How to handle the alignment.

    Returns
    -------
    npd.NestedFrame
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if aligned_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(
            left,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=left_catalog_info.healpix_order,
        )

    right_joined_df = right_meta if right is None else concat_partition_and_margin(right, right_margin)

    right_joined_df = pack_flat(npd.NestedFrame(right_joined_df.set_index(right_on))).rename(right_name)

    merged = left.reset_index().merge(right_joined_df, left_on=left_on, right_index=True, how=how)
    merged.set_index(left_catalog_info.healpix_column, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument, too-many-locals
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
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
):
    """Performs a join on two catalog partitions through an association catalog

    Parameters
    ----------
    left : npd.NestedFrame
        The left partition to merge
    right : npd.NestedFrame
        The right partition to merge
    right_margin : npd.NestedFrame
        The right margin partition to merge
    through : npd.NestedFrame
        The association column partition to merge with
    left_pixel : HealpixPixel
        The HEALPix pixel of the left partition
    right_pixel : HealpixPixel
        The HEALPix pixel of the right partition
    right_margin_pixel : HealpixPixel
        The HEALPix pixel of the right margin partition
    through_pixel : HealpixPixel
        The HEALPix pixel of the association partition
    left_catalog_info : hc.TableProperties
        The hats structure of the left catalog
    right_catalog_info : hc.TableProperties
        The hats structure of the right catalog
    right_margin_catalog_info : hc.TableProperties
        The hats structure of the right margin catalog
    assoc_catalog_info : hc.TableProperties
        The hats structure of the association catalog
    suffixes : tuple[str, str]
        The suffixes to apply to each partition's column names
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

    Returns
    -------
    npd.NestedFrame
        A dataframe with the result of merging the left and right partitions on the specified columns
    """
    if assoc_catalog_info.primary_column is None or assoc_catalog_info.join_column is None:
        raise ValueError("Invalid catalog_info")
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(
            left, right_pixel.order, right_pixel.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    right_joined_df = concat_partition_and_margin(right, right_margin)

    left_join_column = apply_left_suffix(
        assoc_catalog_info.primary_column, right_joined_df.columns, suffixes, suffix_method
    )
    right_join_column = apply_right_suffix(
        assoc_catalog_info.join_column, left.columns, suffixes, suffix_method
    )
    left, right_joined_df = apply_suffixes(left, right_joined_df, suffixes, suffix_method, log_changes=False)

    # Edge case: if right_column + suffix == join_column_association, columns will be in the wrong order
    # so rename association column
    join_column_association = assoc_catalog_info.join_column_association
    if join_column_association in right_joined_df.columns:
        join_column_association = join_column_association + "_assoc"
        through.rename(
            columns={assoc_catalog_info.join_column_association: join_column_association}, inplace=True
        )

    join_columns = [assoc_catalog_info.primary_column_association, join_column_association]
    join_columns_to_drop = []
    for c in join_columns:
        if c not in left.columns and c not in right_joined_df.columns and c not in join_columns_to_drop:
            join_columns_to_drop.append(c)

    cols_to_drop = [c for c in NON_JOINING_ASSOCIATION_COLUMNS if c in through.columns]
    if len(cols_to_drop) > 0:
        through = through.drop(cols_to_drop, axis=1)

    merged = (
        left.reset_index()
        .merge(
            through,
            left_on=left_join_column,
            right_on=assoc_catalog_info.primary_column_association,
        )
        .merge(
            right_joined_df,
            left_on=join_column_association,
            right_on=right_join_column,
        )
    )

    extra_join_cols = through.columns.drop(join_columns + cols_to_drop)
    other_cols = merged.columns.drop(extra_join_cols)

    merged = merged[other_cols.append(extra_join_cols)]

    merged.set_index(left_catalog_info.healpix_column, inplace=True)
    if len(join_columns_to_drop) > 0:
        merged.drop(join_columns_to_drop, axis=1, inplace=True)
    return merged


# pylint: disable=too-many-arguments, unused-argument
def perform_merge_asof(
    left: npd.NestedFrame,
    right: npd.NestedFrame,
    left_pixel: HealpixPixel,
    right_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    suffixes: tuple[str, str],
    direction: str,
    suffix_method: str | None = None,
):
    """Performs a merge_asof on two catalog partitions

    Parameters
    ----------
    left : npd.NestedFrame
        The left partition to merge
    right : npd.NestedFrame
        The right partition to merge
    left_pixel : HealpixPixel
        The HEALPix pixel of the left partition
    right_pixel : HealpixPixel
        The HEALPix pixel of the right partition
    left_catalog_info : hc.TableProperties
        The catalog info of the left catalog
    right_catalog_info : hc.TableProperties
        The catalog info of the right catalog
    suffixes : Tuple[str
        The suffixes to apply to each partition's column names
    direction : str
        The direction to perform the merge_asof
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

    Returns
    -------
    npd.NestedFrame
        A dataframe with the result of merging the left and right partitions on the
        specified columns with `merge_asof`
    """
    if right_pixel.order > left_pixel.order:
        left = filter_by_spatial_index_to_pixel(
            left, right_pixel.order, right_pixel.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    left, right = apply_suffixes(left, right, suffixes, suffix_method, log_changes=False)
    left.sort_index(inplace=True)
    right.sort_index(inplace=True)
    merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction=direction)
    return merged


def join_catalog_data_on(
    left: Catalog,
    right: Catalog,
    left_on: str,
    right_on: str,
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
    log_changes: bool = True,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs spatially on a specified column

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to join
    right : lsdb.Catalog
        The right catalog to join
    left_on : str
        The column to join on from the left partition
    right_on : str
        The column to join on from the right partition
    suffixes : tuple[str, str]
        The suffixes to apply to each partition's column names
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

        .. warning:: This default will change to "overlapping_columns" in a future release.

    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'.

    Returns
    -------
    npd.NestedFrame
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
        suffix_method,
    )

    meta_df = generate_meta_df_for_joined_tables(
        (left, right), suffixes, suffix_method=suffix_method, log_changes=log_changes
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def join_catalog_data_nested(
    left: Catalog,
    right: Catalog,
    left_on: str,
    right_on: str,
    nested_column_name: str | None = None,
    how: str = "inner",
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs spatially on a specified column, adding the right as a nested column with nested
    dask

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to join
    right : lsdb.Catalog
        The right catalog to join
    left_on : str
        The column to join on from the left partition
    right_on : str
        The column to join on from the right partition
    nested_column_name : str
        The name of the nested column in the final output, if None, defaults to
        name of the right catalog
    how : One of {'left', 'inner'}, default 'inner'
        How to handle the alignment (Default: 'inner').

    Returns
    -------
    npd.NestedFrame
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    if how not in ["inner", "left"]:
        raise ValueError("`how` needs to be 'inner' or 'left'")

    alignment_type = PixelAlignmentType(how)

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    if nested_column_name is None:
        nested_column_name = right.name

    alignment = align_catalogs(left, right, alignment_type=alignment_type)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels), (None, aligned_pixels)],
        perform_join_nested,
        left_on,
        right_on,
        nested_column_name,
        right._ddf._meta,  # pylint: disable=protected-access
        how,
    )

    meta_df = generate_meta_df_for_nested_tables([left], right, nested_column_name, join_column_name=right_on)

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def join_catalog_data_through(
    left: Catalog,
    right: Catalog,
    association: AssociationCatalog,
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
    log_changes: bool = True,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Joins two catalogs with an association table

    Parameters
    ----------
    left : lsdb.Catalog
        the left catalog to join
    right : lsdb.Catalog
        the right catalog to join
    association : AssociationCatalog
        the association catalog to join the catalogs with
    suffixes : Tuple[str
        the suffixes to apply to each partition's column names
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

        .. warning:: This default will change to "overlapping_columns" in a future release.
    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'.

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
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
            suffix_method=suffix_method,
            log_changes=log_changes,
        )

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )
    elif association.max_separation is None:
        warnings.warn(
            "Association catalog does not specify maximum separation."
            " Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )
    elif right.margin.hc_structure.catalog_info.margin_threshold < association.max_separation:
        warnings.warn(
            f"Right catalog margin threshold ({right.margin.hc_structure.catalog_info.margin_threshold})"
            f" is smaller than association maximum separation ({association.max_separation})."
            " Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    alignment = align_catalogs_with_association(left, association, right)
    left_pixels, assoc_pixels, right_pixels = get_healpix_pixels_from_association(alignment)

    joined_partitions = align_and_apply(
        [
            (left, left_pixels),
            (right, right_pixels),
            (right.margin, right_pixels),
            (association, assoc_pixels),
        ],
        perform_join_through,
        suffixes,
        suffix_method,
    )

    association_join_columns = [
        association.hc_structure.catalog_info.primary_column_association,
        association.hc_structure.catalog_info.join_column_association,
    ]
    non_joining_columns = [c for c in NON_JOINING_ASSOCIATION_COLUMNS if c in association.columns]

    # pylint: disable=protected-access
    extra_df = association._ddf._meta.drop(non_joining_columns + association_join_columns, axis=1)
    meta_df = generate_meta_df_for_joined_tables(
        (left, right), suffixes, extra_columns=extra_df, suffix_method=suffix_method, log_changes=log_changes
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def merge_asof_catalog_data(
    left: Catalog,
    right: Catalog,
    suffixes: tuple[str, str],
    direction: str = "backward",
    suffix_method: str | None = None,
    log_changes: bool = True,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Uses the pandas `merge_asof` function to merge two catalogs on their indices by distance of keys

    Must be along catalog indices, and does not include margin caches, meaning results may be incomplete for
    merging points.

    This function is intended for use in special cases such as Dust Map Catalogs, for general merges,
    the `crossmatch`and `join` functions should be used.

    Parameters
    ----------
    left : lsdb.Catalog
        the left catalog to join
    right : lsdb.Catalog
        the right catalog to join
    suffixes : Tuple[str
        the suffixes to apply to each partition's column names
    direction : str
        the direction to perform the merge_asof
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs

        .. warning:: This default will change to "overlapping_columns" in a future release.
    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'.

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
        A tuple of the dask dataframe with the result of the join, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    alignment = align_catalogs(left, right)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels)], perform_merge_asof, suffixes, direction, suffix_method
    )

    meta_df = generate_meta_df_for_joined_tables(
        (left, right), suffixes, suffix_method=suffix_method, log_changes=log_changes
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)
