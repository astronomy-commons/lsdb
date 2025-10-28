from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Type

from hats.pixel_tree import PixelAlignment

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
    generate_meta_df_for_joined_tables,
    generate_meta_df_for_nested_tables,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


# pylint: disable=too-many-arguments, unused-argument
def perform_crossmatch(
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
    suffixes,
    suffix_method,
    meta_df,
    **kwargs,
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
    algorithm : BuiltInCrossmatchAlgorithm | Callable
        The algorithm to use to perform the
        crossmatch. Can be specified using a string for a built-in algorithm, or a custom
        method. For more details, see `crossmatch` method in the `Catalog` class.
    suffixes : tuple[str,str] | None
        The suffixes to append to the column names from the left and right catalogs respectively
    suffix_method : str, default 'all_columns'
        Method to use to add suffixes to columns. Options are:

        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.
    **kwargs
        Additional arguments to pass to the crossmatching algorithm.

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions.
    """
    if right_pix.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df, right_pix.order, right_pix.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    if len(left_df) == 0:
        return meta_df

    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)

    return algorithm(
        left_df,
        right_joined_df,
        left_pix.order,
        left_pix.pixel,
        right_pix.order,
        right_pix.pixel,
        left_catalog_info,
        right_catalog_info,
        right_margin_catalog_info,
    ).crossmatch(suffixes, suffix_method=suffix_method, **kwargs)


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
    **kwargs,
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
    algorithm : BuiltInCrossmatchAlgorithm | Callable
        The algorithm to use to perform the
        crossmatch. Can be specified using a string for a built-in algorithm, or a custom
        method. For more details, see `crossmatch` method in the `Catalog` class.
    nested_column_name : str
        The name of the nested column in the resulting dataframe storing the
        joined columns in the right catalog. (Default: name of right catalog)
    meta_df : npd.NestedFrame
        The final meta for the crossmatch.
    **kwargs
        Additional arguments to pass to the crossmatching algorithm.

    Returns
    -------
    npd.NestedFrame
        DataFrame with the results of crossmatching for the pair of partitions. The results
        are stored in a nested column.
    """
    if right_pix.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(
            left_df, right_pix.order, right_pix.pixel, spatial_index_order=left_catalog_info.healpix_order
        )

    if len(left_df) == 0:
        return meta_df

    right_joined_df = concat_partition_and_margin(right_df, right_margin_df)

    return algorithm(
        left_df,
        right_joined_df,
        left_pix.order,
        left_pix.pixel,
        right_pix.order,
        right_pix.pixel,
        left_catalog_info,
        right_catalog_info,
        right_margin_catalog_info,
    ).crossmatch_nested(nested_column_name, **kwargs)


# pylint: disable=too-many-locals
def crossmatch_catalog_data(
    left: Catalog,
    right: Catalog,
    suffixes: tuple[str, str],
    algorithm: (
        Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
    ) = BuiltInCrossmatchAlgorithm.KD_TREE,
    suffix_method: str | None = None,
    log_changes: bool = True,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs

    Parameters
    ----------
    left : lsdb.Catalog
        the left catalog to perform the cross-match on
    right : lsdb.Catalog
        the right catalog to perform the cross-match on
    suffixes : tuple[str,str]
        the suffixes to append to the column names from the left and
        right catalogs respectively
    algorithm : BuiltInCrossmatchAlgorithm | Callable
        The algorithm to use to perform the
        crossmatch. Can be specified using a string for a built-in algorithm, or a custom
        method. For more details, see `crossmatch` method in the `Catalog` class.
        (Default value = BuiltInCrossmatchAlgorithm.KD_TREE)
    suffix_method : str | None, default 'all_columns'
        Method to use to add suffixes to columns. Options are:
        
        - "overlapping_columns": only add suffixes to columns that are present in both catalogs
        - "all_columns": add suffixes to all columns from both catalogs
    log_changes : bool, default True
        If True, logs an info message for each column that is being renamed.
        This only applies when suffix_method is 'overlapping_columns'. Default: True
    **kwargs :
        Additional arguments to pass to the cross-match algorithm

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input catalogs.
    """
    crossmatch_algorithm = get_crossmatch_algorithm(algorithm)
    # Create an instance of the crossmatch algorithm, using the metadata dataframes
    # and the provided kwargs.
    crossmatch_algorithm.validate(left, right, **kwargs)

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
    meta_df = generate_meta_df_for_joined_tables(
        (left, right),
        suffixes,
        suffix_method=suffix_method,
        extra_columns=crossmatch_algorithm.extra_columns,
        log_changes=log_changes,
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels)],
        perform_crossmatch,
        crossmatch_algorithm,
        suffixes,
        suffix_method,
        meta_df,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


# pylint: disable=too-many-locals
def crossmatch_catalog_data_nested(
    left: Catalog,
    right: Catalog,
    nested_column_name: str,
    algorithm: (
        Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
    ) = BuiltInCrossmatchAlgorithm.KD_TREE,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Cross-matches the data from two catalogs with the result from the right catalog in a nested column

    Parameters
    ----------
    left : lsdb.Catalog
        the left catalog to perform the cross-match on
    right : lsdb.Catalog
        the right catalog to perform the cross-match on
    nested_column_name : str
        The name of the nested column that will contain the crossmatched rows
        from the right catalog
    algorithm : BuiltInCrossmatchAlgorithm | Callable
        The algorithm to use to perform the
        crossmatch. Can be specified using a string for a built-in algorithm, or a custom
        method. For more details, see `crossmatch` method in the `Catalog` class.
        (Default value = BuiltInCrossmatchAlgorithm.KD_TREE)
    **kwargs
        Additional arguments to pass to the cross-match algorithm

    Returns
    -------
    tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]
        A tuple of the dask dataframe with the result of the cross-match, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    crossmatch_algorithm = get_crossmatch_algorithm(algorithm)
    # Create an instance of the crossmatch algorithm, using the metadata dataframes
    # and the provided kwargs.
    crossmatch_algorithm.validate(left, right, **kwargs)

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
        [left], right, nested_column_name, extra_nested_columns=crossmatch_algorithm.extra_columns
    )

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels), (right.margin, right_pixels)],
        perform_crossmatch_nested,
        crossmatch_algorithm,
        nested_column_name,
        meta_df,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)


def get_crossmatch_algorithm(
    algorithm: type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm,
) -> type[AbstractCrossmatchAlgorithm]:
    """Gets the function to perform a cross-match algorithm

    Parameters
    ----------
    algorithm : type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
        The algorithm to use to perform the cross-match. Can be specified using a string
        for a built-in algorithm, or a custom method.

    Returns
    -------
    type[AbstractCrossmatchAlgorithm]
        The function to perform the specified crossmatch. Either by looking up the method for a
        built-in algorithm, or returning the custom function.
    """
    if isinstance(algorithm, BuiltInCrossmatchAlgorithm):
        return builtin_crossmatch_algorithms[algorithm]
    if issubclass(algorithm, AbstractCrossmatchAlgorithm):
        return algorithm
    raise TypeError("algorithm must be either callable or a string for a builtin algorithm")
