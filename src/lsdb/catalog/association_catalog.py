import hats as hc
import nested_dask as nd

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap


class AssociationCatalog(HealpixDataset):
    """LSDB Association Catalog DataFrame to perform join analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hats.AssociationCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.AssociationCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)
