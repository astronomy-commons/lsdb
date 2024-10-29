from __future__ import annotations

from typing import Any, Callable, Dict

import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
from dask import delayed
from hats.pixel_math import HealpixPixel, spatial_index_to_healpix


@delayed
def perform_inner_skymap(
    partition: npd.NestedFrame,
    func: Callable[[npd.NestedFrame, HealpixPixel], Any],
    pixel: HealpixPixel,
    target_order: int,
    default_value: Any = 0,
    **kwargs,
) -> np.ndarray:
    """Splits a partition into pixels at a target order and performs a given function on the new pixels"""
    delta_order = target_order - pixel.order
    img = np.full(1 << 2 * delta_order, fill_value=default_value)

    if len(partition) == 0:
        return img

    spatial_index = partition.index.to_numpy()
    order_pixels = spatial_index_to_healpix(spatial_index, target_order=target_order)

    def apply_func(df):
        # gets the healpix pixel of the partition using the spatial index
        p = spatial_index_to_healpix([df.index.to_numpy()[0]], target_order=target_order)[0]
        return func(df, HealpixPixel(target_order, p), **kwargs)

    gb = partition.groupby(order_pixels, sort=False).apply(apply_func)
    min_pixel_value = pixel.pixel << 2 * delta_order
    img[gb.index.to_numpy() - min_pixel_value] = gb.to_numpy(na_value=default_value)
    return img


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
    if len(pixels) == 0:
        npix = hp.order2npix(order) if order is not None else hp.order2npix(0)
        return np.full(npix, default_value)
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
        if value is not None:
            img[np.arange(start, end)] = value

    for s, e, v in zip(starts, ends, values):
        set_values(s, e, v)

    return img
