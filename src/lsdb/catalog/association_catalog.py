import dask.dataframe as dd
import hipscat as hc

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap


class AssociationCatalog(HealpixDataset):
    """LSDB Association Catalog DataFrame to perform join analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hipscat.AssociationCatalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
        self,
        ddf: dd.core.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.AssociationCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)
