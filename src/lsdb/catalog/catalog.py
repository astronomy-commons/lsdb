from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.dask.crossmatch_catalog_data import (
    crossmatch_catalog_data,
    CrossmatchAlgorithmType,
)

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

    def crossmatch(
        self,
        other: Catalog,
        suffixes: Tuple[str, str] | None = None,
        algorithm: CrossmatchAlgorithmType
        | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE,
        name: str | None = None,
        **kwargs,
    ) -> Catalog:
        """Perform a cross-match between two catalogs

        The pixels from each catalog are aligned via a `PixelAlignment`, and cross-matching is
        performed on each pair of overlapping pixels. The resulting catalog will have partitions
        matching an inner pixel alignment - using pixels that have overlap in both input catalogs
        and taking the smallest of any overlapping pixels.

        The resulting catalog will be partitioned using the left catalog's ra and dec, and the
        index for each row will be the same as the index from the corresponding row in the left
        catalog's index.

        Args:
            other (Catalog): The right catalog to cross-match against
            suffixes (Tuple[str, str]): A pair of suffixes to be appended to the end of each column
                name when they are joined. Default: uses the name of the catalog for the suffix
            algorithm (BuiltInCrossmatchAlgorithm | ufunc): The algorithm to use to perform the
                crossmatch. Can be either a string to specify one of the built-in cross-matching
                methods, or a function with a custom method.

                Built-in methods:
                    -`kd_tree`: find the k-nearest neighbors using a kd_tree

                Custom function:
                    The function should be able to perform a crossmatch on two pandas DataFrames
                    from a HEALPix pixel from each catalog. It should return a dataframe with the
                    combined set of columns from the input dataframes with the appropriate suffixes,
                    and a column `_DIST` with the distance between the points.

                    The signature of the function should be:
                    crossmatch(
                        left: pd.DataFrame,
                        right: pd.DataFrame,
                        left_order: int,
                        left_pixel: int,
                        right_order: int,
                        right_pixel: int,
                        left_metadata: hc.catalog.Catalog,
                        right_metadata: hc.catalog.Catalog,
                        suffixes: Tuple[str, str],
                        **kwargs:
                    )
            name (str): The name of the resulting catalog. Default: {left_name}_x_{right_name}

        Returns:
            A Catalog with the data from the left and right catalogs merged with one row for each
            pair of neighbors found from cross-matching.

            The resulting table contains all columns from the left and right catalogs with their
            respective suffixes, and a `_DIST` column with the great circle separation between the
            points.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")
        if name is None:
            name = f"{self.name}_x_{other.name}"
        ddf, ddf_map, alignment = crossmatch_catalog_data(
            self, other, suffixes, algorithm=algorithm, **kwargs
        )
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            catalog_name=name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)
