from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Type

from hats.pixel_tree import PixelAlignment

import pandas as pd

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
def perform_concat(
    left_df,
    right_df,
    left_pix,
    right_pix,
    left_catalog_info,
    right_catalog_info,
    **kwargs,
):
    """Performs a crossmatch on data from a HEALPix pixel in each catalog

    Filters the left catalog before performing the cross-match to stop duplicate points appearing in
    the result.
    """
    if right_pix.order > left_pix.order:
        left_df = filter_by_spatial_index_to_pixel(left_df, right_pix.order, right_pix.pixel)

    if left_pix.order > right_pix.order:
        right_df = filter_by_spatial_index_to_pixel(right_df, left_pix.order, left_pix.pixel)

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
    alignment = align_catalogs(left, right, add_right_margin=False)

    # get lists of HEALPix pixels from alignment to pass to cross-match
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    # generate meta table structure for dask df
    meta_df = pd.concat([left._ddf._meta, right._ddf._meta], **kwargs)

    # perform the crossmatch on each partition pairing using dask delayed for lazy computation
    joined_partitions = align_and_apply(
        [(left, left_pixels), (right, right_pixels)],
        perform_concat,
        **kwargs,
    )

    return construct_catalog_args(joined_partitions, meta_df, alignment)