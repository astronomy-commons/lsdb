import hats as hc

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


class MapCatalog(HealpixDataset):
    """LSDB DataFrame to contain a continuous map.

    Attributes:
        hc_structure: `hats.MapCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.MapCatalog
