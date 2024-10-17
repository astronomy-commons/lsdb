from typing import Dict, TypeVar

from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset

DaskDFPixelMap = Dict[HealpixPixel, int]

CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)
HCCatalogTypeVar = TypeVar("HCCatalogTypeVar", bound=HCHealpixDataset)
