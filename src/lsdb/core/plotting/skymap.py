from typing import Any, Dict

import healpy as hp
import numpy as np
from hipscat.pixel_math import HealpixPixel


def plot_skymap(pixel_map: Dict[HealpixPixel, Any], **kwargs):
    """Plot a map of healpix_pixels with values on a mollweide projection.

    Args:
        pixel_map(Dict[HealpixPixel, Any]): A dictionary of healpix pixels and their values
        kwargs: Additional keyword arguments to pass to `healpy.mollview`
    """

    pixels = list(pixel_map.keys())
    hp_orders = np.vectorize(lambda x: x.order)(pixels)
    hp_pixels = np.vectorize(lambda x: x.pixel)(pixels)
    max_order = np.max(hp_orders)
    npix = hp.order2npix(max_order)
    img = np.zeros(npix)
    dorders = max_order - hp_orders
    values = np.vectorize(lambda x: pixel_map[x])(pixels)
    starts = hp_pixels * (4**dorders)
    ends = (hp_pixels + 1) * (4**dorders)

    def set_values(start, end, value):
        img[np.arange(start, end)] = value

    np.vectorize(set_values)(starts, ends, values)

    hp.mollview(img, nest=True, **kwargs)
