from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Type

import pandas as pd
from hats.pixel_tree import PixelAlignment, PixelAlignmentType

import lsdb.nested as nd
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import (
    BuiltInCrossmatchAlgorithm,
    builtin_crossmatch_algorithms,
)
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    filter_by_spatial_index_to_margin,
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_healpix_pixels_from_alignment,
    concat_align_catalogs,
    get_aligned_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


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
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if left_df is None:
        left_df = aligned_meta
    if right_df is None:
        right_df = aligned_meta

    if left_pix is not None and aligned_pix.order > left_pix.order and left_df is not None:
        left_df = filter_by_spatial_index_to_pixel(left_df, aligned_pix.order, aligned_pix.pixel)

    if right_pix is not None and aligned_pix.order > right_pix.order and right_df is not None:
        right_df = filter_by_spatial_index_to_pixel(right_df, aligned_pix.order, aligned_pix.pixel)

    return pd.concat([left_df, right_df], **kwargs)


# pylint: disable=too-many-locals
def concat_catalog_data(
    left: Catalog,
    right: Catalog,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Args:
        left (lsdb.Catalog): the left catalog to perform the cross-match on
        right (lsdb.Catalog): the right catalog to perform the cross-match on
        suffixes (Tuple[str,str]): the suffixes to append to the column names from the left and
            right catalogs respectively
        algorithm (BuiltInCrossmatchAlgorithm | Callable): The algorithm to use to perform the
            crossmatch. Can be specified using a string for a built-in algorithm, or a custom
            method. For more details, see `crossmatch` method in the `Catalog` class.
        **kwargs: Additional arguments to pass to the cross-match algorithm

    Returns:
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """

    # perform alignment on the two catalogs
    alignment = concat_align_catalogs(
        left,
        right,
        filter_by_mocs=True,
        alignment_type=PixelAlignmentType.OUTER,
    )

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = pd.concat([left._ddf._meta, right._ddf._meta], **kwargs)

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (None, aligned_pixels)],
        perform_concat,
        aligned_meta=meta_df,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


# pylint: disable=too-many-arguments, unused-argument
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
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if left_pix is None:
        output_margin_df = None
        if right_pix.order == aligned_pix.order:
            output_margin_df = right_margin_df
        else:
            combined_right_df = concat_partition_and_margin(right_df, right_margin_df)
            output_margin_df = filter_by_spatial_index_to_margin(
                combined_right_df,
                aligned_pix.order,
                aligned_pix.pixel,
                margin_radius,
            )
        return pd.concat([output_margin_df, aligned_meta], **kwargs)

    if right_pix is None:
        output_margin_df = None
        if left_pix.order == aligned_pix.order:
            output_margin_df = left_margin_df
        else:
            combined_left_df = concat_partition_and_margin(left_df, left_margin_df)
            output_margin_df = filter_by_spatial_index_to_margin(
                combined_left_df,
                aligned_pix.order,
                aligned_pix.pixel,
                margin_radius,
            )
        return pd.concat([output_margin_df, aligned_meta], **kwargs)

    if right_pix.order > left_pix.order:
        combined_left_df = concat_partition_and_margin(left_df, left_margin_df)
        filtered_left_df = filter_by_spatial_index_to_margin(
            combined_left_df,
            right_pix.order,
            right_pix.pixel,
            margin_radius,
        )
        return pd.concat(
            [filtered_left_df, right_margin_df if right_margin_df is not None else aligned_meta], **kwargs
        )

    if left_pix.order > right_pix.order:
        combined_right_df = concat_partition_and_margin(right_df, right_margin_df)
        filtered_right_df = filter_by_spatial_index_to_margin(
            combined_right_df,
            left_pix.order,
            left_pix.pixel,
            margin_radius,
        )
        return pd.concat(
            [left_margin_df if left_margin_df is not None else aligned_meta, filtered_right_df], **kwargs
        )

    return pd.concat(
        [
            left_margin_df if left_margin_df is not None else aligned_meta,
            right_margin_df if right_margin_df is not None else aligned_meta,
        ],
        **kwargs,
    )


# pylint: disable=too-many-locals
def concat_margin_data(
    left: Catalog,
    right: Catalog,
    margin_radius: float,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Args:
        left (lsdb.Catalog): the left catalog to perform the cross-match on
        right (lsdb.Catalog): the right catalog to perform the cross-match on
        suffixes (Tuple[str,str]): the suffixes to append to the column names from the left and
            right catalogs respectively
        algorithm (BuiltInCrossmatchAlgorithm | Callable): The algorithm to use to perform the
            crossmatch. Can be specified using a string for a built-in algorithm, or a custom
            method. For more details, see `crossmatch` method in the `Catalog` class.
        **kwargs: Additional arguments to pass to the cross-match algorithm

    Returns:
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """

    # perform alignment on the two catalogs
    alignment = concat_align_catalogs(
        left,
        right,
        filter_by_mocs=False,
        alignment_type=PixelAlignmentType.OUTER,
    )

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = pd.concat([left._ddf._meta, right._ddf._meta], **kwargs)

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
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
