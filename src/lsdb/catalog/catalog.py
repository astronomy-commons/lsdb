from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
from hipscat.pixel_math import HealpixPixel
import numpy as np

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data
from lsdb.dask.join_catalog_data import join_catalog_data


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
        if suffixes is None:
            suffixes = ("", "")
        ddf, ddf_map, alignment = join_catalog_data(self, other, through, suffixes=suffixes)
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def crossmatch(self, other: Catalog, suffixes: Tuple[str, str] | None = None) -> Catalog:
        if suffixes is None:
            suffixes = ("", "")
        ddf, ddf_map, alignment = crossmatch_catalog_data(self, other, suffixes)
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def query(self, qarg: str=None) -> Catalog:
        if qarg is None:
            raise Exception("Must pass a string query argument like: 'column_name1 > 0'")
        ddf = self._ddf.query(qarg)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)

    def where(self, qarg: str) -> Catalog:
        return self.query(qarg=qarg)
    
    def assign(self, **kwargs) -> Catalog:
        if len(kwargs) == 0 or len(kwargs) > 1:
            raise Exception("Invalid assigning of column. Must be a single lambda function")
        ddf = self._ddf.assign(**kwargs)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)
    
    def for_each(self, ufunc, **kwargs) -> Catalog:
        ddf = self._ddf.groupby("_hipscat_index").apply(ufunc, **kwargs)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)
