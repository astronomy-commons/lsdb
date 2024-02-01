import dask.dataframe as dd
import hipscat as hc

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap


class MarginCatalog(HealpixDataset):
    """LSDB Catalog DataFrame to contain the "margin" of another HiPSCat catalog.
    spatial operations.

    Attributes:
        hc_structure: `hipscat.MarginCatalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: hc.catalog.MarginCatalog

    def __init__(
        self,
        ddf: dd.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.MarginCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)
