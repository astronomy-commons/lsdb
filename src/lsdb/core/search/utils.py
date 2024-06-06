from typing import List, Tuple, cast

import dask.dataframe as dd
import pandas as pd
from dask.delayed import Delayed
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import HCCatalog


def _perform_search(
    ddf: dd.DataFrame,
    ddf_pixel_map: dict,
    metadata: HCCatalog,
    search: AbstractSearch,
) -> Tuple[dd.core.DataFrame, dict]:
    """Performs a search on the catalog from a list of pixels to search in.

    Args:
        ddf (dd.DataFrame): The catalog Dask DataFrame to be filtered.
        ddf_pixel_map (dict): The catalog pixel-to-partition map.
        metadata (HCCatalog): The metadata of the hipscat catalog.
        search (AbstractSearch): Instance of AbstractSearch.

    Returns:
        A tuple with a dask dataframe containing the search results and a dictionary
        mapping pixel to partition index.
    """
    partitions = ddf.to_delayed()
    filtered_pixels = metadata.get_healpix_pixels()
    targeted_partitions = [partitions[ddf_pixel_map[pixel]] for pixel in filtered_pixels]
    filtered_partitions = (
        [search.search_points(partition, metadata.catalog_info) for partition in targeted_partitions]
        if search.fine
        else targeted_partitions
    )
    # pylint: disable=protected-access
    return _construct_search_ddf(filtered_pixels, filtered_partitions, ddf._meta)


def _construct_search_ddf(
    filtered_pixels: List[HealpixPixel], filtered_partitions: List[Delayed], meta: pd.DataFrame
) -> Tuple[dd.core.DataFrame, dict]:
    """Constructs a Dask Dataframe and respective catalog pixel map from the list of
    pixels and respective delayed partitions.

    Args:
        filtered_pixels (List[HealpixPixel]): The list of pixels in the search
        filtered_partitions (List[Delayed]): The list of delayed partitions
        meta (pd.DataFrame): The metadata to use for the Dask DataFrame

    Returns:
        The Dask DataFrame and respective catalog pixel map.
    """
    divisions = get_pixels_divisions(filtered_pixels)
    ddf = dd.io.from_delayed(filtered_partitions, meta=meta, divisions=divisions)
    ddf = cast(dd.core.DataFrame, ddf)
    ddf_partition_map = {pixel: i for i, pixel in enumerate(filtered_pixels)}
    return ddf, ddf_partition_map
