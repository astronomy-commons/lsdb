from typing import List

from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_HEALPIX_ORDER


def get_ordered_pixel_list(pixels: List[HealpixPixel]) -> List[HealpixPixel]:
    """Sort pixels by pixel number at highest order. This depth-first
    approach allows to get the pixels in monotonically ascending order.

    Args:
        pixels (List[HealpixPixel]): The list of catalog HEALPix pixels

    Returns:
        The list of HEALPix pixels sorted by pixel number, in monotonically
        ascending order.
    """
    sorted_pixels = sorted(
        pixels,
        key=lambda pixel: (4 ** (HIPSCAT_ID_HEALPIX_ORDER - pixel.order)) * pixel.pixel,
    )
    return sorted_pixels
