from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data, CrossmatchAlgorithmType

DaskDFPixelMap = Dict[HealpixPixel, int]


# pylint: disable=R0903, W0212
class Catalog(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
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
        super().__init__(ddf, hc_structure)
        self._ddf_pixel_map = ddf_pixel_map

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

    @property
    def name(self):
        return self.hc_structure.catalog_name

    def crossmatch(self,
                   other: Catalog,
                   suffixes: Tuple[str, str] | None = None,
                   algorithm: CrossmatchAlgorithmType | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE,
                   **kwargs,
                   ) -> Catalog:
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")
        ddf, ddf_map, alignment = crossmatch_catalog_data(
            self,
            other,
            suffixes,
            algorithm=algorithm,
            **kwargs)
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            catalog_name=f"{self.name}_x_{other.name}",
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)
