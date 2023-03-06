from abc import ABC, abstractmethod
from typing import Tuple

import dask.dataframe as dd
import hipscat as hc

from lsdb.catalog.catalog import DaskDFPixelMap


class AbstractCatalogLoader(ABC):
    """Interface to load a HiPSCat catalog."""

    @abstractmethod
    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        """Load a `hipscat.Catalog` object with catalog metadata and pixel
        structure."""

    @abstractmethod
    def load_dask_df(self) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
        """Load a Dask DataFrame with the catalog source data, and the map of
        HEALPix pixels to partitions."""

    @abstractmethod
    def get_source_info_dict(self) -> dict:
        """Get a dictionary with the parameters used to load the catalog"""
