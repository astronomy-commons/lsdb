# pylint: disable=duplicate-code
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple

import dask
import nested_dask as nd
import nested_pandas as npd
import pandas as pd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment

from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    align_catalogs,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


# pylint: disable=too-many-arguments, unused-argument
@dask.delayed
def perform_merge_map(
    catalog_partition: npd.NestedFrame,
    map_partition: npd.NestedFrame,
    catalog_pixel: HealpixPixel,
    map_pixel: HealpixPixel,
    catalog_structure: TableProperties,
    map_structure: TableProperties,
    func: Callable[..., npd.NestedFrame],
    *args,
    **kwargs,
):
    """Performs a merge_asof on two catalog partitions

    Args:
        catalog_partition (npd.NestedFrame): the left partition to merge
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
    if map_pixel.order > catalog_pixel.order:
        catalog_partition = filter_by_spatial_index_to_pixel(
            catalog_partition, map_pixel.order, map_pixel.pixel
        )

    catalog_partition.sort_index(inplace=True)
    map_partition.sort_index(inplace=True)
    return func(catalog_partition, map_partition, catalog_pixel, map_pixel, *args, **kwargs)


def merge_map_catalog_data(
    point_catalog: Catalog,
    map_catalog: Catalog,
    func: Callable[..., npd.NestedFrame],
    *args,
    meta: pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None = None,
    **kwargs,
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

    alignment = align_catalogs(point_catalog, map_catalog)

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    partitions_with_func = align_and_apply(
        [(point_catalog, left_pixels), (map_catalog, right_pixels)],
        perform_merge_map,
        func,
        *args,
        **kwargs,
    )

    return construct_catalog_args(partitions_with_func, meta, alignment)
