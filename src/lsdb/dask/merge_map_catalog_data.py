# pylint: disable=duplicate-code
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import nested_dask as nd
import nested_pandas as npd
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment, PixelAlignmentType
from hats.pixel_tree.pixel_alignment import align_with_mocs

from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    construct_catalog_args,
    filter_by_spatial_index_to_pixel,
    get_healpix_pixels_from_alignment,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog import Catalog, MapCatalog


# pylint: disable=too-many-arguments, unused-argument
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
    """Applies a function to each pair of partitions in this catalog and the map catalog.

    Args:
        catalog_partition (npd.NestedFrame): partition of the point-source catalog
        map_partition (npd.NestedFrame): partition of the continuous map catalog
        catalog_pixel (HealpixPixel): the HEALPix pixel of the catalog partition
        map_pixel (HealpixPixel): the HEALPix pixel of the map partition
        catalog_structure (hc.TableProperties): the catalog info of the catalog
        map_structure (hc.TableProperties): the catalog info of the map
        func (Callable): method to apply to the two partitions

    Returns:
        A dataframe with the result of calling `func`
    """
    if map_pixel.order > catalog_pixel.order:
        catalog_partition = filter_by_spatial_index_to_pixel(
            catalog_partition, map_pixel.order, map_pixel.pixel
        )

    catalog_partition.sort_index(inplace=True)
    map_partition.sort_index(inplace=True)
    return func(catalog_partition, map_partition, catalog_pixel, map_pixel, *args, **kwargs)


# pylint: disable=protected-access
def merge_map_catalog_data(
    point_catalog: Catalog,
    map_catalog: MapCatalog,
    func: Callable[..., npd.NestedFrame],
    *args,
    meta: npd.NestedFrame | None = None,
    **kwargs,
) -> tuple[nd.NestedFrame, DaskDFPixelMap, PixelAlignment]:
    """Applies a function to each pair of partitions in this catalog and the map catalog.

    The pixels from each catalog are aligned via a `PixelAlignment`, and the respective dataframes
    are passed to the function. The resulting catalog will have the same partitions as the point
    source catalog.

    Args:
        point_catalog (lsdb.Catalog): the point-source catalog to apply
        map_catalog (lsdb.MapCatalog): the continuous map catalog to apply
        func (Callable): The function applied to each catalog partition, which will be called with:
            `func(catalog_partition: npd.NestedFrame, map_partition: npd.NestedFrame, `
            ` healpix_pixel: HealpixPixel, *args, **kwargs)`
            with the additional args and kwargs passed to the `merge_map` function.
        *args: Additional positional arguments to call `func` with.
        meta (pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None): An empty pandas DataFrame that
            has columns matching the output of the function applied to the catalog partition. Other types
            are accepted to describe the output dataframe format, for full details see the dask
            documentation https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
            If meta is None (default), LSDB will try to work out the output schema of the function by
            calling the function with an empty DataFrame. If the function does not work with an empty
            DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
            will generate empty partitions, though these can be removed by calling the
            `Catalog.prune_empty_partitions` method.
        **kwargs: Additional keyword args to pass to the function. These are passed to the Dask DataFrame
            `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
            `transform_divisions` will be passed through and work as described in the dask documentation
            https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html


    Returns:
        A tuple of the dask dataframe with the result of the operation, the pixel map from HEALPix
        pixel to partition index within the dataframe, and the PixelAlignment of the two input
        catalogs.
    """
    if meta is None:
        meta = func(
            point_catalog._ddf._meta.copy(),
            map_catalog._ddf._meta.copy(),
            HealpixPixel(0, 0),
            HealpixPixel(0, 0),
        )
        if meta is None:
            raise ValueError(
                "func returned None for empty DataFrame input. The function must return a value, changing"
                " the partitions in place will not work. If the function does not work for empty inputs, "
                "please specify a `meta` argument."
            )
    alignment = align_with_mocs(
        point_catalog.hc_structure.pixel_tree,
        map_catalog.hc_structure.pixel_tree,
        point_catalog.hc_structure.moc,
        map_catalog.hc_structure.moc,
        alignment_type=PixelAlignmentType.INNER,
    )

    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)

    partitions_with_func = align_and_apply(
        [(point_catalog, left_pixels), (map_catalog, right_pixels)],
        perform_merge_map,
        func,
        *args,
        **kwargs,
    )

    return construct_catalog_args(partitions_with_func, meta, alignment)
