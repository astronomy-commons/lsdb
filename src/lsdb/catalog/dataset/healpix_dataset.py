import warnings
from typing import List, Tuple, cast

import dask.dataframe as dd
import numpy as np
from dask.delayed import Delayed
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from typing_extensions import Self

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import DaskDFPixelMap


# pylint: disable=W0212
class HealpixDataset(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hipscat.Dataset` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: HCHealpixDataset

    def __init__(
        self,
        ddf: dd.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: HCHealpixDataset,
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

    def get_healpix_pixels(self) -> List[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog

        Returns:
            List of all Healpix pixels in the catalog
        """
        return self.hc_structure.get_healpix_pixels()

    def get_ordered_healpix_pixels(self) -> List[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog,
        ordered by breadth-first nested ordering.

        Returns:
            List of all Healpix pixels in the catalog
        """
        pixels = self.get_healpix_pixels()
        return np.array(pixels)[get_pixel_argsort(pixels)]

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

    def query(self, expr: str) -> Self:
        """Filters catalog using a complex query expression

        Args:
            expr (str): Query expression to evaluate. The column names that are not valid Python
                variables names should be wrapped in backticks, and any variable values can be
                injected using f-strings. The use of '@' to reference variables is not supported.
                More information about pandas query strings is available
                `here <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`__.

        Returns:
            A catalog that contains the data from the original catalog that complies
            with the query expression
        """
        ddf = self._ddf.query(expr)
        return self.__class__(ddf, self._ddf_pixel_map, self.hc_structure)

    def _perform_search(self, filtered_pixels: List[HealpixPixel], search: AbstractSearch):
        """Performs a search on the catalog from a list of pixels to search in

        Args:
            filtered_pixels (List[HealpixPixel]): List of pixels in the catalog to be searched
            search (AbstractSearch): The search object to perform the search with

        Returns:
            A tuple containing a dictionary mapping pixel to partition index and a dask dataframe
            containing the search results
        """
        partitions = self._ddf.to_delayed()
        targeted_partitions = [partitions[self._ddf_pixel_map[pixel]] for pixel in filtered_pixels]
        filtered_partitions = [search.search_points(partition) for partition in targeted_partitions]
        return self._construct_search_ddf(filtered_pixels, filtered_partitions)

    def _construct_search_ddf(self, filtered_pixels, filtered_partitions):
        """Constructs the search Dask DataFrame and the respective pixel map"""
        divisions = get_pixels_divisions(filtered_pixels)
        search_ddf = dd.from_delayed(filtered_partitions, meta=self._ddf._meta, divisions=divisions)
        search_ddf = cast(dd.DataFrame, search_ddf)
        ddf_partition_map = {pixel: i for i, pixel in enumerate(filtered_pixels)}
        return ddf_partition_map, search_ddf

    def prune_empty_partitions(self) -> Self:
        """Removes empty partitions from a catalog

        Returns:
            A new catalog containing only the non-empty partitions
        """
        warnings.warn("Pruning empty partitions is expensive. It may run slow!", RuntimeWarning)
        non_empty_pixels, non_empty_partitions = self._get_non_empty_partitions()
        ddf_partition_map, search_ddf = self._construct_search_ddf(non_empty_pixels, non_empty_partitions)
        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(non_empty_pixels)
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)

    def _get_non_empty_partitions(self) -> Tuple[List[HealpixPixel], List[Delayed]]:
        """Computes the partition lengths and returns the indices of those that are empty"""
        partitions = self._ddf.to_delayed()
        # Compute partition lengths (expensive operation)
        partition_sizes = self._ddf.map_partitions(len).compute()
        empty_partition_indices = np.argwhere(partition_sizes == 0).flatten()
        # Extract the non-empty pixels and respective partitions
        non_empty_pixels, non_empty_partitions = [], []
        for pixel, partition_index in self._ddf_pixel_map.items():
            if partition_index not in empty_partition_indices:
                non_empty_pixels.append(pixel)
                non_empty_partitions.append(partitions[partition_index])
        return non_empty_pixels, non_empty_partitions
