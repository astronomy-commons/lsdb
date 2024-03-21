from __future__ import annotations

from typing import Any, Callable, Dict

import healpy as hp
import numpy as np
import pandas as pd
from dask import delayed
from hipscat.pixel_math import HealpixPixel

from lsdb.dask.merge_catalog_functions import filter_by_hipscat_index_to_pixel


@delayed
def perform_inner_skymap(
    partition: pd.DataFrame,
    func: Callable[[pd.DataFrame, HealpixPixel], Any],
    pixel: HealpixPixel,
    target_order: int,
    **kwargs,
) -> np.ndarray:
    """Splits a partition into pixels at a target order and performs a given function on the new pixels"""
    delta_order = target_order - pixel.order
    pixels = np.arange(pixel.pixel << (2 * delta_order), (pixel.pixel + 1) << (2 * delta_order))
    return np.vectorize(
        lambda p: func(
            filter_by_hipscat_index_to_pixel(partition, target_order, p),
            HealpixPixel(target_order, p),
            **kwargs,
        )
    )(pixels)


def compute_skymap(
    pixel_map: Dict[HealpixPixel, Any], order: int | None = None, default_value: Any = 0.0
) -> np.ndarray:
    """Returns a histogram map of healpix_pixels to values.

    Args:
        pixel_map(Dict[HealpixPixel, Any]): A dictionary of healpix pixels and their values
        order (int): The order to make the histogram at (default None, uses max order in pixel_map)
        default_value: The value to use at pixels that aren't covered by the pixel_map (default 0)
    """

    pixels = list(pixel_map.keys())
    hp_orders = np.vectorize(lambda x: x.order)(pixels)
    hp_pixels = np.vectorize(lambda x: x.pixel)(pixels)
    if order is None:
        order = np.max(hp_orders)
    npix = hp.order2npix(order)
    img = np.full(npix, default_value)
    dorders = order - hp_orders
    values = [pixel_map[x] for x in pixels]
    starts = hp_pixels << (2 * dorders)
    ends = (hp_pixels + 1) << (2 * dorders)

    def set_values(start, end, value):
        img[np.arange(start, end)] = value

    for s, e, v in zip(starts, ends, values):
        set_values(s, e, v)

    return img
