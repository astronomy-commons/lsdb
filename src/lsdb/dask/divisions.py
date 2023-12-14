from __future__ import annotations

from typing import List, Tuple

import numpy as np
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_MAX, healpix_to_hipscat_id


def get_pixels_divisions(healpix_pixels: List[HealpixPixel]) -> Tuple[int, ...] | None:
    """Calculates the Dask Dataframe divisions for a list of HEALPix pixels.

    Divisions include the minimum value of every HEALPix pixel hipscat_id
    and the maximum value of the last HEALPix pixel hipscat_id. In practice
    they represent the bounds of hipscat_id values for the target pixels.

    Args:
        healpix_pixels (List[HealpixPixel]): The list of HEALPix pixels to
            calculate the hipscat_id bounds for.

    Returns:
        The Dask Dataframe divisions, as a tuple of integer.
    """
    if len(healpix_pixels) == 0:
        return None
    orders = [pix.order for pix in healpix_pixels]
    pixels = [pix.pixel for pix in healpix_pixels]
    divisions = healpix_to_hipscat_id(orders, pixels)
    divisions = np.append(divisions, HIPSCAT_ID_MAX)
    return tuple(np.sort(divisions))
