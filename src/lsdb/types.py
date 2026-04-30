from typing import TypeVar

from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset

HCCatalogTypeVar = TypeVar("HCCatalogTypeVar", bound=HCHealpixDataset)
