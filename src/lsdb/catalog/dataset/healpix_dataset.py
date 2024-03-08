from typing import Any, Callable, Dict, List, cast

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.delayed import Delayed, delayed
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from typing_extensions import Self

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.plotting.skymap import plot_skymap
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

    def _perform_search(self, filtered_pixels: List[HealpixPixel], search: AbstractSearch, fine: bool = True):
        """Performs a search on the catalog from a list of pixels to search in

        Args:
            filtered_pixels (List[HealpixPixel]): List of pixels in the catalog to be searched.
            search (AbstractSearch): Instance of AbstractSearch.
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A tuple containing a dictionary mapping pixel to partition index and a dask dataframe
            containing the search results
        """
        partitions = self._ddf.to_delayed()
        targeted_partitions = [partitions[self._ddf_pixel_map[pixel]] for pixel in filtered_pixels]
        filtered_partitions = (
            [search.search_points(partition) for partition in targeted_partitions]
            if fine
            else targeted_partitions
        )
        divisions = get_pixels_divisions(filtered_pixels)
        search_ddf = dd.from_delayed(filtered_partitions, meta=self._ddf._meta, divisions=divisions)
        search_ddf = cast(dd.DataFrame, search_ddf)
        ddf_partition_map = {pixel: i for i, pixel in enumerate(filtered_pixels)}
        return ddf_partition_map, search_ddf

    def skymap_data(
        self, func: Callable[[pd.DataFrame, HealpixPixel], Any], **kwargs
    ) -> Dict[HealpixPixel, Delayed]:
        """Perform a function on each partition of the catalog, returning a dict of values for each pixel.

        Args:
            func (Callable[[pd.DataFrame, HealpixPixel], Any]): A function that takes a pandas
                DataFrame with the data in a partition, the HealpixPixel of the partition, and any other
                keyword arguments and returns an aggregated value
            **kwargs: Arguments to pass to the function

        Returns:
            A dict of Delayed values, one for the function applied to each partition of the catalog
        """

        partitions = self.to_delayed()
        results = {
            pixel: delayed(func)(partitions[index], pixel, **kwargs)
            for pixel, index in self._ddf_pixel_map.items()
        }
        return results

    def skymap(self, func: Callable[[pd.DataFrame, HealpixPixel], Any], **kwargs):
        """Plot a skymap of an aggregate function applied over each partition with a Mollweide projection.

        Args:
            func (Callable[[pd.DataFrame], HealpixPixel, Any]): A function that takes a pandas DataFrame and
                the HealpixPixel the partition is from and returns a value
            **kwargs: Arguments to pass to healpy.mollview function
        """

        smdata = self.skymap_data(func, **kwargs)
        pixels = list(smdata.keys())
        results = dask.compute(*[smdata[pixel] for pixel in pixels])
        result_dict = {pixels[i]: results[i] for i in range(len(pixels))}
        plot_skymap(result_dict)
