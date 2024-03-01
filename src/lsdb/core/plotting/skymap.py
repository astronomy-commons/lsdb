from typing import Dict, Any

import healpy as hp
import numpy as np

from hipscat.pixel_math import HealpixPixel


def plot_skymap(pixel_map: Dict[HealpixPixel, Any], **kwargs):
    max_order = max(pixel_map.keys(), key=lambda x: x.order).order
    npix = hp.order2npix(max_order)
    img = np.zeros(npix)
    for pixel, value in pixel_map.items():
        dorder = max_order - pixel.order
        start = pixel.pixel * (4 ** dorder)
        end = (pixel.pixel + 1) * (4 ** dorder)
        img_order_pixels = np.arange(start, end)
        img[img_order_pixels] = value
    hp.mollview(img, nest=True, **kwargs)
