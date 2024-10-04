from __future__ import annotations

from typing import List, Tuple

import numpy as np
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import healpix_to_spatial_index


def get_pixels_divisions(healpix_pixels: List[HealpixPixel]) -> Tuple[int, ...] | None:
    """Calculates the Dask Dataframe divisions for a list of HEALPix pixels.

    Divisions include the minimum value of every HEALPix pixel spatial_index
    and the maximum value of the last HEALPix pixel spatial_index. In practice
    they represent the bounds of spatial_index values for the target pixels.

    Args:
        healpix_pixels (List[HealpixPixel]): The list of HEALPix pixels to
            calculate the spatial_index bounds for.

    Returns:
        The Dask Dataframe divisions, as a tuple of integer.
    """
    if len(healpix_pixels) == 0:
        return None
    orders = [pix.order for pix in healpix_pixels]
    pixels = [pix.pixel for pix in healpix_pixels]
    divisions = healpix_to_spatial_index(orders, pixels)
    last_pixel = healpix_pixels[get_pixel_argsort(healpix_pixels)[-1]]
    divisions = np.append(divisions, healpix_to_spatial_index(last_pixel.order, last_pixel.pixel + 1))
    return tuple(np.sort(divisions))
