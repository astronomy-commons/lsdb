from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment, PixelAlignmentType

import lsdb.nested as nd
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    concat_partition_and_margin,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_aligned_pixels_from_alignment,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


# pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument, too-many-locals
def perform_crossmatch(
    left_df: npd.NestedFrame,
    right_df: npd.NestedFrame,
    right_margin_df: npd.NestedFrame,
    aligned_df: npd.NestedFrame | None,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    right_margin_pix: HealpixPixel,
    aligned_pixel: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    right_margin_catalog_info: TableProperties,
    aligned_catalog_info: TableProperties | None,
    algorithm: AbstractCrossmatchAlgorithm,
    how: str,
    suffixes: tuple[str, str],
    suffix_method: str,
    meta_df: npd.NestedFrame,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.

    Parameters
    ----------
    left_df : npd.NestedFrame | None
        Partition from the left catalog.
    right_df : npd.NestedFrame | None
        Partition from the right catalog.
    right_margin_df: npd.NestedFrame | None
        Partition from the right catalog margin cache.
    aligned_df : npd.NestedFrame | None
        The partition of the aligned pixel
    left_pix : HealpixPixel | None
        HealpixPixel for the left partition.
    right_pix : HealpixPixel | None
        HealpixPixel for the right partition.
    right_margin_pix : HealpixPixel | None
        HealpixPixel for the right margin partition.
    aligned_pixel : HealpixPixel
        HealpixPixel for the aligned partition.
    left_catalog_info : hats.catalog.TableProperties
        Catalog info for the left partition.
    right_catalog_info : hats.catalog.TableProperties
        Catalog info for the right partition.
    right_margin_catalog_info : hats.catalog.TableProperties
        Catalog info for the right margin partition.
    aligned_catalog_info : hats.catalog.TableProperties | None
        Catalog info for the aligned partition; usually None
    algorithm : AbstractCrossmatchAlgorithm
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how : str
        How to handle the crossmatch of the two catalogs.
        One of {'left', 'inner'}.
    suffixes : tuple[str,str] | None
        The suffixes to append to the column names from the left and right catalogs respectively
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions.
    """
    # If there's no left partition for this aligned pixel, return an empty/meta frame
    if left_df is None or left_df.size == 0:
        return meta_df
    # The aligned_pixel will be the right_pix if the pixels orders are already
    # compatible, that is, it's the smaller of the left and right pixels.
    if aligned_pixel.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df,
            aligned_pixel.order,
            aligned_pixel.pixel,
            spatial_index_order=left_catalog_info.healpix_order,
        )

    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)
    # If there is no right data for this aligned pixel, provide an empty dataframe with the
    # expected RA/DEC columns so the algorithm can be instantiated safely.
    if right_joined_df is None:
        # Prefer constructing an empty pandas-backed NestedFrame whose columns and dtypes
        # match the *original* right-catalog columns (pre-suffix). We derive the dtypes
        # from the provided `meta_df`, which contains the suffixed column names and
        # expected dtypes for the final joined table.
        right_suffix = suffixes[1] if suffixes is not None else None
        cols: dict = {}
        if right_suffix is not None:
            # meta_df is a nested-pandas NestedFrame with suffixed column names
            # Find all columns that belong to the right side by suffix and map them
            # back to their unsuffixed original names and dtypes.
            suffixed_cols = [c for c in meta_df.columns if c.endswith(right_suffix)]
            for sc in suffixed_cols:
                orig = sc[: -len(right_suffix)]
                # Use the dtype from meta_df so the partition meta will match
                cols[orig] = pd.Series(dtype=meta_df[sc].dtype)
        else:
            # Fallback: use RA/DEC column names if suffixes are not available
            ra_col = getattr(right_catalog_info, "ra_column", None)
            dec_col = getattr(right_catalog_info, "dec_column", None)
            if ra_col:
                cols[ra_col] = pd.Series(dtype=float)
            if dec_col:
                cols[dec_col] = pd.Series(dtype=float)

        # Create a plain nested-pandas NestedFrame (pandas-backed) for the empty partition
        # so that algorithms that expect concrete partition data (pandas NestedFrame) work.
        right_joined_df = npd.NestedFrame(pd.DataFrame(cols))

    crossmatch_args = CrossmatchArgs(
        left_df=left_df,
        right_df=right_joined_df,
        left_order=left_pix.order,
        left_pixel=left_pix.pixel,
        right_order=right_pix.order,
        right_pixel=right_pix.pixel,
        left_catalog_info=left_catalog_info,
        right_catalog_info=right_catalog_info,
        right_margin_catalog_info=right_margin_catalog_info,
    )
    return algorithm.crossmatch(crossmatch_args, how, suffixes, suffix_method)


# pylint: disable=too-many-arguments, unused-argument
def perform_crossmatch_nested(
    left_df,
    right_df,
    right_margin_df,
    left_pix,
    right_pix,
    right_margin_pix,
    left_catalog_info,
    right_catalog_info,
    right_margin_catalog_info,
    algorithm,
    nested_column_name,
    meta_df,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog with result in a nested column

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.

    Parameters
    ----------
    left_df : npd.NestedFrame | None
        Partition from the left catalog.
    right_df : npd.NestedFrame | None
        Partition from the right catalog.
    right_margin_df: npd.NestedFrame | None
        Partition from the right catalog margin cache.
    left_pix : HealpixPixel | None
        HealpixPixel for the left partition.
    right_pix : HealpixPixel | None
        HealpixPixel for the right partition.
    right_margin_pix : HealpixPixel | None
        HealpixPixel for the right margin partition.
    left_catalog_info : hats.catalog.TableProperties
        Catalog info for the left partition.
    right_catalog_info : hats.catalog.TableProperties
        Catalog info for the right partition.
    right_margin_catalog_info : hats.catalog.TableProperties
        Catalog info for the right margin partition.
    algorithm : AbstractCrossmatchAlgorithm
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    nested_column_name : str
        The name of the nested column in the resulting dataframe storing the
        joined columns in the right catalog. (Default: name of right catalog)
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions.
        The results are stored in a nested column.
    """
    # If there's no left partition for this aligned pixel, return an empty/meta frame
    if left_df is None or left_df.size == 0:
        return meta_df

    if right_pix.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df, right_pix.order, right_pix.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)

    crossmatch_args = CrossmatchArgs(
        left_df=left_df,
        right_df=right_joined_df,
        left_order=left_pix.order,
        left_pixel=left_pix.pixel,
        right_order=right_pix.order,
        right_pixel=right_pix.pixel,
        left_catalog_info=left_catalog_info,
        right_catalog_info=right_catalog_info,
        right_margin_catalog_info=right_margin_catalog_info,
    )
    return algorithm.crossmatch_nested(crossmatch_args, nested_column_name)


# pylint: disable=too-many-locals
def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    algorithm: AbstractCrossmatchAlgorithm,
    how: str,
    suffixes: tuple[str, str],
    suffix_method: str | None = None,
    log_changes: bool = True,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to perform the cross-match on
    right : lsdb.Catalog
        The right catalog to perform the cross-match on
    algorithm : AbstractCrossmatchAlgorithm, default `KDTreeCrossmatch`
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    how: str
        How to handle the crossmatch of the two catalogs.
        One of {'left', 'inner'}.
    suffixes : tuple[str,str]
        The suffixes to append to the column names from the left and
        right catalogs respectively.
    suffix_method : str | None, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'. Default: True

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input catalogs.
    """
    # Validate the algorithm parameters
    algorithm.validate(left, right)

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    # perform alignment on the two catalogs
    alignment = align_catalogs(
        left, right, add_right_margin=True, alignment_type=PixelAlignmentType[how.upper()]
    )
    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    aligned_pixels = get_aligned_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_joined_tables(
        (left, right),
        suffixes,
        suffix_method=suffix_method,
        extra_columns=algorithm.extra_columns,
        log_changes=log_changes,
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels), (None, aligned_pixels)],
        perform_crossmatch,
        algorithm,
        how,
        suffixes,
        suffix_method,
        meta_df,
    )

    nf, pixel_map, alignment = construct_catalog_args(joined_partitions, meta_df, alignment)

    return nf, pixel_map, alignment


# pylint: disable=too-many-locals
def crossmatch_catalog_data_nested(
    left: Catalog,
    right: Catalog,
    algorithm: AbstractCrossmatchAlgorithm,
    nested_column_name: str,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Crossmatches the data from two catalogs with the result from the right catalog in a nested column

    Parameters
    ----------
    left : lsdb.Catalog
        The left catalog to perform the cross-match on
    right : lsdb.Catalog
        The right catalog to perform the cross-match on
    algorithm : AbstractCrossmatchAlgorithm, default `KDTreeCrossmatch`
        The algorithm to use to perform the crossmatch. Specified by subclassing
        `AbstractCrossmatchAlgorithm`. For more details, see `crossmatch` method
        in the `Catalog` class.
    nested_column_name : str
        The name of the nested column that will contain the crossmatched rows
        from the right catalog

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input catalogs.
    """
    # Validate the algorithm parameters
    algorithm.validate(left, right)

    if right.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    # perform alignment on the two catalogs
    alignment = align_catalogs(left, right, add_right_margin=True)

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = generate_meta_df_for_nested_tables(
        [left], right, nested_column_name, extra_nested_columns=algorithm.extra_columns
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels)],
        perform_crossmatch_nested,
        algorithm,
        nested_column_name,
        meta_df,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)
