from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.dataframe.join_catalog_data import join_catalog_data, crossmatch_catalog_data

if TYPE_CHECKING:
    from lsdb.catalog.association_catalog.association_catalog import \
        AssociationCatalog

DaskDFPixelMap = Dict[HealpixPixel, int]


# pylint: disable=R0903, W0212
class Catalog(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        name: Name of the catalog
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    def __init__(
        self,
        ddf: dd.core.DataFrame,
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
        super().__init__(ddf, hc_structure)
        self._ddf_pixel_map = ddf_pixel_map

    def get_partition(self, order: int, pixel: int) -> dd.core.DataFrame:
        """Get the dask partition for a given HEALPix pixel

        Args:
            order: Order of HEALPix pixel
            pixel: HEALPix pixel number in NESTED ordering scheme
        Returns:
            Dask Dataframe with a single partition with data at that pixel
        Raises:
            Value error if no data exists for the specified pixel
        """
        partition_index = self.get_partition_index(order, pixel)
        return self._ddf.partitions[partition_index]

    def get_partition_index(self, order: int, pixel: int) -> int:
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
        return partition_index

    def join(self, other: Catalog, through: AssociationCatalog=None, suffixes: Tuple[str, str] | None = None) -> Catalog:
        if through is None:
            raise NotImplementedError("must specify through association catalog")
        ddf, ddf_map, alignment = join_catalog_data(self, other, through, suffixes=suffixes)
        hc_catalog = hc.catalog.Catalog(self.hc_structure.catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def crossmatch(self, other: Catalog, suffixes: Tuple[str, str] | None = None) -> Catalog:
        ddf, ddf_map, alignment = crossmatch_catalog_data(self, other, suffixes)
        hc_catalog = hc.catalog.Catalog(self.hc_structure.catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)
