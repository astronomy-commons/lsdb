from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from hats.pixel_tree import PixelAlignment, PixelAlignmentType

import lsdb.nested as nd
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    concat_align_catalogs,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_spatial_index_to_margin,
    filter_by_spatial_index_to_pixel,
    get_aligned_pixels_from_alignment,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


def _check_strict_column_types(meta1: pd.DataFrame, meta2: pd.DataFrame):
    """
    Raises a TypeError if columns with the same name have different dtypes.

    Args:
        meta1 (pd.DataFrame): First DataFrame to compare.
        meta2 (pd.DataFrame): Second DataFrame to compare.

    Raises:
        TypeError: If any columns with the same name have conflicting dtypes.

    Notes:
        This function is useful for ensuring strict schema consistency when concatenating
        or merging DataFrames. It checks only columns present in both DataFrames.
    """
    for col in set(meta1.columns) & set(meta2.columns):
        dtype1 = meta1[col].dtype
        dtype2 = meta2[col].dtype
        if dtype1 != dtype2:
            raise TypeError(f"Column '{col}' has conflicting dtypes: {dtype1} (left) vs {dtype2} (right)")


def _reindex_and_coerce_dtypes(
    df: pd.DataFrame | None,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reindex DataFrame columns and coerce dtypes to match a reference
    meta DataFrame.

    Args:
        df (pd.DataFrame | None):
            DataFrame to be reindexed and coerced. If None, returns meta.
        meta (pd.DataFrame):
            Reference DataFrame whose columns and dtypes should be
            matched.

    Returns:
        pd.DataFrame:
            DataFrame with columns and dtypes matching `meta`. Missing
            columns are filled with NA values and correct dtype.

    Raises:
        TypeError:
            If dtype conversion fails for any column.

    Notes:
        - This function is useful to ensure all partitions in a Dask
          DataFrame have consistent schema.
    """
    if df is None:
        df = meta.copy()
    else:
        df = df.reindex(columns=meta.columns, copy=False)
    for col in meta.columns:
        try:
            df[col] = df[col].astype(meta[col].dtype, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Could not convert column '{col}' to dtype {meta[col].dtype}: {e}") from e
    return df


def _is_all_na(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame contains only null values or is empty.

    Args:
        df (pd.DataFrame): DataFrame to check.

    Returns:
        bool: True if all values are null or DataFrame is empty, False otherwise.
    """
    if df.size == 0:
        return True
    return not df.notna().any().any()


def _concat_meta_safe(meta: pd.DataFrame, parts: list[pd.DataFrame | None], **kwargs) -> pd.DataFrame:
    """
    Concatenate only non-empty and non-all-NA DataFrames, preserving meta schema and dtypes.

    Args:
        meta (pd.DataFrame): Reference DataFrame for schema and dtypes.
        parts (list[pd.DataFrame | None]): List of DataFrames to concatenate.
        **kwargs: Additional keyword arguments for pandas.concat.

    Returns:
        pd.DataFrame: Concatenated DataFrame with schema and dtypes matching meta.

    Notes:
        - All output columns will match the order and dtypes of meta.
        - Missing columns are filled with NA values and correct dtype.
        - Empty result returns meta.iloc[0:0].copy() coerced to meta dtypes.
    """
    keep: list[pd.DataFrame] = []
    for p in parts:
        if p is None:
            continue
        if len(p) == 0:
            continue
        if _is_all_na(p):
            continue
        keep.append(p)

    if not keep:
        return _reindex_and_coerce_dtypes(meta.iloc[0:0].copy(), meta)

    if len(keep) == 1:
        return _reindex_and_coerce_dtypes(keep[0], meta)

    # Reindex and coerce all kept DataFrames to meta before concatenation
    keep = [_reindex_and_coerce_dtypes(df, meta) for df in keep]
    out = pd.concat(keep, **kwargs)
    return _reindex_and_coerce_dtypes(out, meta)


# pylint: disable=too-many-arguments, unused-argument
def perform_concat(
    left_df,
    right_df,
    aligned_df,
    left_pix,
    right_pix,
    aligned_pix,
    left_catalog_info,
    right_catalog_info,
    aligned_catalog_info,
    aligned_meta,
    **kwargs,
):
    """
    Concatenate partitions for a single aligned pixel from two catalogs.

    Args:
        left_df (pd.DataFrame | None): Partition from the left catalog.
        right_df (pd.DataFrame | None): Partition from the right catalog.
        aligned_df (pd.DataFrame | None): Partition for the aligned pixel.
        left_pix: HealpixPixel for the left partition.
        right_pix: HealpixPixel for the right partition.
        aligned_pix: HealpixPixel for the aligned partition.
        left_catalog_info: Catalog info for the left partition.
        right_catalog_info: Catalog info for the right partition.
        aligned_catalog_info: Catalog info for the aligned partition.
        aligned_meta (pd.DataFrame): Meta DataFrame for column order and dtypes.
        **kwargs: Additional keyword arguments for pandas.concat.

    Returns:
        pd.DataFrame: Concatenated DataFrame for the aligned pixel.
    """
    # Filter to aligned pixel when needed (handles order differences)
    if left_pix is not None and aligned_pix.order > left_pix.order and left_df is not None:
        left_df = filter_by_spatial_index_to_pixel(left_df, aligned_pix.order, aligned_pix.pixel)

    if right_pix is not None and aligned_pix.order > right_pix.order and right_df is not None:
        right_df = filter_by_spatial_index_to_pixel(right_df, aligned_pix.order, aligned_pix.pixel)

    # Substitute None with meta to preserve schema
    if left_df is None:
        left_df = aligned_meta
    if right_df is None:
        right_df = aligned_meta

    # Normalize column order
    left_df = _reindex_and_coerce_dtypes(left_df, aligned_meta)
    right_df = _reindex_and_coerce_dtypes(right_df, aligned_meta)

    # Concatenate without empty/all-NA inputs to avoid FutureWarning
    return _concat_meta_safe(aligned_meta, [left_df, right_df], **kwargs)


# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments, unused-argument
def perform_margin_concat(
    left_df,
    left_margin_df,
    right_df,
    right_margin_df,
    aligned_df,
    left_pix,
    left_margin_pix,
    right_pix,
    right_margin_pix,
    aligned_pix,
    left_catalog_info,
    left_margin_catalog_info,
    right_catalog_info,
    right_margin_catalog_info,
    aligned_catalog_info,
    margin_radius,
    aligned_meta,
    **kwargs,
):
    """
    Concatenate margin partitions for a single aligned pixel from two catalogs.

    Args:
        left_df (pd.DataFrame | None): Partition from the left catalog.
        left_margin_df (pd.DataFrame | None): Margin partition from the left catalog.
        right_df (pd.DataFrame | None): Partition from the right catalog.
        right_margin_df (pd.DataFrame | None): Margin partition from the right catalog.
        aligned_df (pd.DataFrame | None): Partition for the aligned pixel.
        left_pix: HealpixPixel for the left partition.
        left_margin_pix: HealpixPixel for the left margin partition.
        right_pix: HealpixPixel for the right partition.
        right_margin_pix: HealpixPixel for the right margin partition.
        aligned_pix: HealpixPixel for the aligned partition.
        left_catalog_info: Catalog info for the left partition.
        left_margin_catalog_info: Catalog info for the left margin partition.
        right_catalog_info: Catalog info for the right partition.
        right_margin_catalog_info: Catalog info for the right margin partition.
        aligned_catalog_info: Catalog info for the aligned partition.
        margin_radius (float): Margin radius in arcseconds.
        aligned_meta (pd.DataFrame): Meta DataFrame for column order and dtypes.
        **kwargs: Additional keyword arguments for pandas.concat.

    Returns:
        pd.DataFrame: Concatenated DataFrame for the aligned pixel margin.
    """
    if left_pix is None:
        # Only right side contributes to this aligned pixel
        if right_pix.order == aligned_pix.order:
            out = right_margin_df
        else:
            combined_right_df = concat_partition_and_margin(right_df, right_margin_df)
            out = filter_by_spatial_index_to_margin(
                combined_right_df, aligned_pix.order, aligned_pix.pixel, margin_radius
            )
        out = _reindex_and_coerce_dtypes(out, aligned_meta)
        return _concat_meta_safe(aligned_meta, [out], **kwargs)

    if right_pix is None:
        # Only left side contributes to this aligned pixel
        if left_pix.order == aligned_pix.order:
            out = left_margin_df
        else:
            combined_left_df = concat_partition_and_margin(left_df, left_margin_df)
            out = filter_by_spatial_index_to_margin(
                combined_left_df, aligned_pix.order, aligned_pix.pixel, margin_radius
            )
        out = _reindex_and_coerce_dtypes(out, aligned_meta)
        return _concat_meta_safe(aligned_meta, [out], **kwargs)

    if right_pix.order > left_pix.order:
        # Right has higher order: filter left (partition ∪ margin) to right's pixel at that order
        combined_left_df = concat_partition_and_margin(left_df, left_margin_df)
        filtered_left_df = filter_by_spatial_index_to_margin(
            combined_left_df, right_pix.order, right_pix.pixel, margin_radius
        )
        left_part = _reindex_and_coerce_dtypes(filtered_left_df, aligned_meta)
        right_part = _reindex_and_coerce_dtypes(right_margin_df, aligned_meta)
        return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)

    if left_pix.order > right_pix.order:
        # Left has higher order: symmetric case to the above
        combined_right_df = concat_partition_and_margin(right_df, right_margin_df)
        filtered_right_df = filter_by_spatial_index_to_margin(
            combined_right_df, left_pix.order, left_pix.pixel, margin_radius
        )
        left_part = _reindex_and_coerce_dtypes(left_margin_df, aligned_meta)
        right_part = _reindex_and_coerce_dtypes(filtered_right_df, aligned_meta)
        return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)

    # Same order on both sides: just stack margins (still normalize)
    left_part = _reindex_and_coerce_dtypes(left_margin_df, aligned_meta)
    right_part = _reindex_and_coerce_dtypes(right_margin_df, aligned_meta)
    return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)


# pylint: disable=too-many-locals
def concat_catalog_data(
    left: Catalog,
    right: Catalog,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """
    Concatenate main catalog data for two catalogs using pixel alignment.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        **kwargs: Additional keyword arguments for pandas.concat.

    Returns:
        tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]: Tuple containing the concatenated
            NestedFrame, pixel map, and pixel alignment.
    """
    # Build alignment across both trees (including margins as pixel trees, but filtered by MOCs)
    alignment = concat_align_catalogs(
        left,
        right,
        filter_by_mocs=True,
        alignment_type=PixelAlignmentType.OUTER,
    )

    # Lists of HEALPix pixels to feed into the map/apply stage
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # Build the meta (union of schemas) with deterministic column order (left then right)
    # pylint: disable=protected-access
    meta_left = left._ddf._meta
    meta_right = right._ddf._meta

    _check_strict_column_types(meta_left, meta_right)

    meta_df = pd.concat([meta_left, meta_right], **kwargs)
    # pylint: enable=protected-access
    # Lazy per-pixel concatenation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (None, aligned_pixels)],
        perform_concat,
        aligned_meta=meta_df,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


# pylint: disable=too-many-locals
def concat_margin_data(
    left: Catalog,
    right: Catalog,
    margin_radius: float,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """
    Concatenate margin data for two catalogs using pixel alignment.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        margin_radius (float): Margin radius in arcseconds.
        **kwargs: Additional keyword arguments for pandas.concat.

    Returns:
        tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]: Tuple containing the concatenated
            NestedFrame, pixel map, and pixel alignment for the margin data.
    """
    # Build alignment across both trees (including margins as pixel trees), no MOC filtering
    alignment = concat_align_catalogs(
        left,
        right,
        filter_by_mocs=False,
        alignment_type=PixelAlignmentType.OUTER,
    )

    # Lists of HEALPix pixels to feed into the map/apply stage
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # Build the meta (union of schemas) with deterministic column order (left then right)
    # pylint: disable=protected-access
    meta_left = left._ddf._meta
    meta_right = right._ddf._meta

    _check_strict_column_types(meta_left, meta_right)

    meta_df = pd.concat([meta_left, meta_right], **kwargs)
    # pylint: enable=protected-access
    # Lazy per-pixel concatenation for margins
    joined_partitions = align_and_apply(
        [
            (left, left_pixels),
            (left.margin, left_pixels),
            (right, right_pixels),
            (right.margin, right_pixels),
            (None, aligned_pixels),
        ],
        perform_margin_concat,
        margin_radius=margin_radius,
        aligned_meta=meta_df,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)
