from typing import Tuple, TypedDict

import hipscat as hc
from dask import dataframe as dd

from lsdb.catalog.catalog import DaskDFPixelMap
from lsdb.catalog.catalog_source_type import CatalogSourceType
from lsdb.loaders.hipscat.abstract_catalog_loader import AbstractCatalogLoader


class LocalSourceInfo(TypedDict):
    """Source information describing how a catalog was loaded from local files

    Attributes:
        source: Type of source data was loaded from
        path: local path of catalog
    """

    source: CatalogSourceType
    path: str


class LocalCatalogLoader(AbstractCatalogLoader):
    """Loads a HiPSCat formatted Catalog from local files"""

    def __init__(self, local_path):
        """Initializes a LocalCatalogLoader

        Args:
            local_path: path to the root of the HiPSCat catalog
        """
        self.path = local_path

    def load_dask_df(self) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
        pass

    def load_hipscat_catalog(self) -> hc.catalog.Catalog:
        pass

    def get_source_info_dict(self) -> LocalSourceInfo:
        """Get source info containing local path of the data

        Returns:
            Source info dictionary
        """
        return {
            "source": CatalogSourceType.LOCAL,
            "path": self.path
        }
