from typing import List, Tuple

from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import healpix_to_hipscat_id


def get_pixels_divisions(ordered_pixels: List[HealpixPixel]) -> Tuple[int, ...]:
    """Calculates the Dask Dataframe divisions for a list of HEALPix pixels.

    Divisions include the minimum value of every HEALPix pixel hipscat_id
    and the maximum value of the last HEALPix pixel hipscat_id. In practice
    they represent the bounds of hipscat_id values for the target pixels.

    Args:
        ordered_pixels (List[HealpixPixel]): The list of HEALPix pixels to
            calculate the hipscat_id bounds for. They must be ordered by
            ascending hipscat_id.

    Returns:
        The Dask Dataframe divisions, as a tuple of integer.
    """
    divisions = []
    for index, hp_pixel in enumerate(ordered_pixels):
        left_bound = healpix_to_hipscat_id(hp_pixel.order, hp_pixel.pixel)
        divisions.append(left_bound)
        if index == len(ordered_pixels) - 1:
            next_order = hp_pixel.order + 1
            next_order_pixel = (hp_pixel.pixel + 1) * 4
            right_bound = healpix_to_hipscat_id(next_order, next_order_pixel)
            divisions.append(right_bound)
    return tuple(divisions)
