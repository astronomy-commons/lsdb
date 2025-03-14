from typing import TypeVar

from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset

DaskDFPixelMap = dict[HealpixPixel, int]

CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)
HCCatalogTypeVar = TypeVar("HCCatalogTypeVar", bound=HCHealpixDataset)
