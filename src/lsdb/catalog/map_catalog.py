import hats as hc

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


class MapCatalog(HealpixDataset):
    """LSDB DataFrame to contain a continuous map."""

    hc_structure: hc.catalog.MapCatalog
