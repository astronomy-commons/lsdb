from lsdb.catalog.dataset.healpix_dataset import HealpixDataset

import hipscat as hc
import dask.dataframe as dd

from lsdb.types import DaskDFPixelMap


class AssociationCatalog(HealpixDataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
        self,
        ddf: dd.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.AssociationCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)
