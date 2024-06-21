from typing import Dict, List, Tuple, TypeVar

from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.pixel_math import HealpixPixel
from typing_extensions import TypeAlias

from lsdb.catalog.dataset.dataset import Dataset

# Compute pixel map returns a tuple. The first element is
# the number of data points within the HEALPix pixel, the
# second element is the list of pixels it contains.
HealpixInfo: TypeAlias = Tuple[int, List[int]]

DaskDFPixelMap = Dict[HealpixPixel, int]

CatalogTypeVar = TypeVar("CatalogTypeVar", bound=Dataset)
HCCatalogTypeVar = TypeVar("HCCatalogTypeVar", bound=HCHealpixDataset)
