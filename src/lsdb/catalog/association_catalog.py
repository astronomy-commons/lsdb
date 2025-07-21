import hats as hc

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


class AssociationCatalog(HealpixDataset):
    """LSDB Association Catalog DataFrame to perform join analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hats.AssociationCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.AssociationCatalog
