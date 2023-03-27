from typing import Dict

import dask.dataframe as dd
import hipscat as hc

from lsdb.core.healpix.healpix_pixel import HealpixPixel

DaskDFPixelMap = Dict[HealpixPixel, int]


# pylint: disable=R0903, W0212
class Catalog:
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        name: Name of the catalog
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    def __init__(
        self,
        ddf: dd.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.Catalog,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
        """
        self._ddf = ddf
        self._ddf_pixel_map = ddf_pixel_map
        self.hc_structure = hc_structure

    def __repr__(self):
        return self._ddf.__repr__()

    def _repr_html_(self):
        return self._ddf._repr_html_()

    def compute(self):
        """Compute dask distributed dataframe to pandas dataframe"""
        return self._ddf.compute()

    def get_partition(self, order: int, pixel: int) -> dd.DataFrame:
        """Get the dask partition for a given HEALPix pixel

        Args:
            order: Order of HEALPix pixel
            pixel: HEALPix pixel number in NESTED ordering scheme
        Returns:
            Dask Dataframe with a single partition with data at that pixel
        Raises:
            Value error if no data exists for the specified pixel
        """
        hp_pixel = HealpixPixel(order, pixel)
        if not hp_pixel in self._ddf_pixel_map:
            raise ValueError(f"Pixel at order {order} pixel {pixel} not in Catalog")
        partition_index = self._ddf_pixel_map[hp_pixel]
        return self._ddf.partitions[partition_index]
