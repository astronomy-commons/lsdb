import dask.dataframe as dd
import hipscat as hc

from lsdb import Catalog
from lsdb.catalog.catalog import DaskDFPixelMap


class AssociationCatalog(Catalog):

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
            self,
            ddf: dd.core.DataFrame,
            ddf_pixel_map: DaskDFPixelMap,
            hc_structure: hc.catalog.AssociationCatalog
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)
