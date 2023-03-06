from typing import Dict, Tuple

import dask.dataframe as dd
import hipscat as hc

DaskDFPixelMap = Dict[Tuple[int, int], int]


# pylint: disable=R0913
# pylint: disable=R0903
class Catalog:
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        name: Name of the catalog
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
        source_info: Data on where the catalog was loaded from
    """

    def __init__(
            self,
            name: str,
            ddf: dd.DataFrame,
            ddf_pixel_map: DaskDFPixelMap,
            hc_structure: hc.catalog.Catalog,
            source_info: dict,
    ):
        """Initialise a Catalog object.

        Args:
            name: Name of the catalog
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
            source_info: dict with source info to get where data was loaded from
        """
        self.name = name
        self._ddf = ddf
        self._ddf_pixel_map = ddf_pixel_map
        self.hc_structure = hc_structure
        self.source_info = source_info
