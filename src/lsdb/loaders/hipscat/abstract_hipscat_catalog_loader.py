from abc import ABC, abstractmethod

from lsdb.catalog.catalog import Catalog
from lsdb.loaders.hipscat.hipscat_loading_config import HipscatLoadingConfig


# pylint: disable=R0903
class AbstractHipscatCatalogLoader(ABC):
    """Interface to load a HiPSCat catalog."""

    @abstractmethod
    def __init__(self, path: str, config: HipscatLoadingConfig) -> None:
        """Initialize a catalog loader with the parameters to load the files"""
        pass

    @abstractmethod
    def load_catalog(self) -> Catalog:
        """Load the `Catalog` object from the parameters the loader was initialized with"""
        pass
