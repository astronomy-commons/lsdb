from typing import TypeVar

from hats.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset

DaskDFPixelMap = dict[HealpixPixel, int]

CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)
