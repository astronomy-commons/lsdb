from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
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


def _concat_no_warn(frames: list[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Concatenate frames while suppressing any pandas FutureWarning (local scope).
    Also defaults to sort=False to avoid accidental column reordering.
    """
    kwargs.setdefault("sort", False)
    with warnings.catch_warnings():
        # Ignore any FutureWarning only within this concat call
        warnings.simplefilter("ignore", category=FutureWarning)
        return pd.concat(frames, **kwargs)


def _reindex_like_meta(df: pd.DataFrame | None, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `df` has the same column order as `meta`. If `df` is None, return `meta`.

    Notes
    -----
    This is critical to keep Dask's `meta` consistent with each partition result.
    Any mismatch in column order between the computed partition and the meta
    will trigger `dask.dataframe.utils.check_meta` errors.
    """
    if df is None:
        return meta
    # Use copy=False to avoid unnecessary copies; columns missing in df become all-NA.
    return df.reindex(columns=meta.columns, copy=False)


def _is_all_na(df: pd.DataFrame) -> bool:
    """
    Return True if the DataFrame has no non-null values (robust to extension dtypes).
    Treat both empty frames and frames with rows but all-null cells as all-NA.
    """
    if df.size == 0:
        return True
    return not df.notna().any().any()


def _concat_meta_safe(meta: pd.DataFrame, parts: list[pd.DataFrame | None], **kwargs) -> pd.DataFrame:
    """
    Safely concatenate only non-empty and non-all-NA parts to avoid pandas' FutureWarning.

    Behavior:
    - If no parts remain, return an empty DataFrame with the `meta` schema.
    - If a single part remains, return it directly (avoid calling pd.concat on length-1 lists).
    - To fully avoid the FutureWarning, temporarily drop columns that are all-NA
      across all kept parts, concat, then reindex back to `meta.columns`, restoring
      those all-NA columns with the exact dtype from `meta`.
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

    # No parts kept → return empty frame with meta schema (and dtypes)
    if not keep:
        return meta.iloc[0:0].copy()

    # Single part → return directly (avoid concat of length-1)
    if len(keep) == 1:
        return keep[0]

    # Identify columns that are all-NA across ALL kept parts.
    # Be tolerant if some part lacks the column (treat as all-NA for that part).
    all_na_cols: list[str] = []
    cols = list(meta.columns)
    for c in cols:
        has_non_na_somewhere = False
        for k in keep:
            if c in k.columns:
                if k[c].notna().any():
                    has_non_na_somewhere = True
                    break
            # else: missing column in this part counts as "all-NA" for that part
        if not has_non_na_somewhere:
            all_na_cols.append(c)

    if all_na_cols:
        # Drop the all-NA columns before concat to avoid pandas' dtype warnings,
        # then reindex and recreate them with the exact dtype from `meta`.
        reduced = [k.drop(columns=all_na_cols, errors="ignore") for k in keep]
        out = _concat_no_warn(reduced, **kwargs)
        out = out.reindex(columns=meta.columns)

        # Recreate dropped columns with precise dtype from `meta`
        for c in all_na_cols:
            target_dtype = meta[c].dtype
            try:
                # Prefer constructing an ExtensionArray with pd.NA and the target dtype
                out[c] = pd.array([pd.NA] * len(out), dtype=target_dtype)
            except (TypeError, ValueError):
                # Fallback: use float NaN and best-effort astype to target dtype
                s = pd.Series([np.nan] * len(out), index=out.index)
                try:
                    out[c] = s.astype(target_dtype, copy=False)
                except (TypeError, ValueError):
                    out[c] = s
        return out

    # Normal path: just concat the kept parts
    return _concat_no_warn(keep, **kwargs)


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
    Per-aligned-pixel concatenation for the *main* (non-margin) tables.

    If an input DF is None, substitute `aligned_meta` to preserve the schema.
    When input pixel orders differ, filter to the aligned pixel region to avoid
    duplication/leakage. Always reindex each piece to `aligned_meta.columns`.
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
    left_df = _reindex_like_meta(left_df, aligned_meta)
    right_df = _reindex_like_meta(right_df, aligned_meta)

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
    Per-aligned-pixel concatenation for the *margin* tables.

    Depending on which side is present and the relative orders, we may need to
    build a combined (partition ∪ margin) table and then filter it to the
    aligned pixel's margin footprint. Each piece is reindexed to `aligned_meta.columns`
    and concatenated using `_concat_meta_safe`.
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
        out = _reindex_like_meta(out, aligned_meta)
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
        out = _reindex_like_meta(out, aligned_meta)
        return _concat_meta_safe(aligned_meta, [out], **kwargs)

    if right_pix.order > left_pix.order:
        # Right has higher order: filter left (partition ∪ margin) to right's pixel at that order
        combined_left_df = concat_partition_and_margin(left_df, left_margin_df)
        filtered_left_df = filter_by_spatial_index_to_margin(
            combined_left_df, right_pix.order, right_pix.pixel, margin_radius
        )
        left_part = _reindex_like_meta(filtered_left_df, aligned_meta)
        right_part = _reindex_like_meta(right_margin_df, aligned_meta)
        return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)

    if left_pix.order > right_pix.order:
        # Left has higher order: symmetric case to the above
        combined_right_df = concat_partition_and_margin(right_df, right_margin_df)
        filtered_right_df = filter_by_spatial_index_to_margin(
            combined_right_df, left_pix.order, left_pix.pixel, margin_radius
        )
        left_part = _reindex_like_meta(left_margin_df, aligned_meta)
        right_part = _reindex_like_meta(filtered_right_df, aligned_meta)
        return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)

    # Same order on both sides: just stack margins (still normalize)
    left_part = _reindex_like_meta(left_margin_df, aligned_meta)
    right_part = _reindex_like_meta(right_margin_df, aligned_meta)
    return _concat_meta_safe(aligned_meta, [left_part, right_part], **kwargs)


# pylint: disable=too-many-locals
def concat_catalog_data(
    left: Catalog,
    right: Catalog,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """
    Concatenate the *main* (non-margin) data for two catalogs over a pixel alignment.

    This builds an OUTER pixel alignment (via `concat_align_catalogs`) and, for each
    aligned pixel, concatenates the partitions from `left` and `right` after normalizing
    their column layout to the `meta` (union of left/right schemas).
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
    meta_df = _concat_no_warn([left._ddf._meta, right._ddf._meta], **kwargs)
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
    Concatenate the *margin* data for two catalogs over a pixel alignment.

    The alignment is OUTER and *not* filtered by MOCs (margins can lie outside MOC).
    For each aligned pixel, we combine the relevant (partition ∪ margin) pieces from
    both sides, optionally filtering to the aligned-pixel's margin footprint when
    orders differ, then normalize to `aligned_meta`.
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
    meta_df = _concat_no_warn([left._ddf._meta, right._ddf._meta], **kwargs)
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
