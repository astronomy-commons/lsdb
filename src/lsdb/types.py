from typing import TypeVar

from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math import HealpixPixel

DaskDFPixelMap = dict[HealpixPixel, int]

HCCatalogTypeVar = TypeVar("HCCatalogTypeVar", bound=HCHealpixDataset)
