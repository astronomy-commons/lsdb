from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Tuple, cast

import dask
import dask.dataframe as dd
import healpy as hp
import hipscat as hc
import numpy as np
import pandas as pd
from dask.delayed import Delayed, delayed
from hipscat.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hipscat.inspection import plot_pixel_list
from hipscat.inspection.visualize_catalog import get_projection_method
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from typing_extensions import Self

from lsdb import io
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.plotting.skymap import compute_skymap, perform_inner_skymap
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
            ValueError: if no data exists for the specified pixel
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
            ValueError: if no data exists for the specified pixel
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

    def _perform_search(
        self,
        metadata: hc.catalog.Catalog,
        filtered_pixels: List[HealpixPixel],
        search: AbstractSearch,
        fine: bool = True,
    ):
        """Performs a search on the catalog from a list of pixels to search in

        Args:
            metadata (hc.catalog.Catalog): The metadata of the hipscat catalog.
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
            [search.search_points(partition, metadata) for partition in targeted_partitions]
            if fine
            else targeted_partitions
        )
        return self._construct_search_ddf(filtered_pixels, filtered_partitions)

    def _construct_search_ddf(
        self, filtered_pixels: List[HealpixPixel], filtered_partitions: List[Delayed]
    ) -> Tuple[dict, dd.DataFrame]:
        """Constructs a search catalog pixel map and respective Dask Dataframe

        Args:
            filtered_pixels (List[HealpixPixel]): The list of pixels in the search
            filtered_partitions (List[Delayed]): The list of delayed partitions

        Returns:
            The catalog pixel map and the respective Dask DataFrame
        """
        divisions = get_pixels_divisions(filtered_pixels)
        search_ddf = dd.from_delayed(filtered_partitions, meta=self._ddf._meta, divisions=divisions)
        search_ddf = cast(dd.DataFrame, search_ddf)
        ddf_partition_map = {pixel: i for i, pixel in enumerate(filtered_pixels)}
        return ddf_partition_map, search_ddf

    def prune_empty_partitions(self, persist: bool = False) -> Self:
        """Prunes the catalog of its empty partitions

        Args:
            persist (bool): If True previous computations are saved. Defaults to False.

        Returns:
            A new catalog containing only its non-empty partitions
        """
        warnings.warn("Pruning empty partitions is expensive. It may run slow!", RuntimeWarning)
        if persist:
            self._ddf.persist()
        non_empty_pixels, non_empty_partitions = self._get_non_empty_partitions()
        ddf_partition_map, search_ddf = self._construct_search_ddf(non_empty_pixels, non_empty_partitions)
        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(non_empty_pixels)
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)

    def _get_non_empty_partitions(self) -> Tuple[List[HealpixPixel], List[Delayed]]:
        """Determines which pixels and partitions of a catalog are not empty

        Returns:
            A tuple with the non-empty pixels and respective partitions
        """
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

    def skymap_data(
        self, func: Callable[[pd.DataFrame, HealpixPixel], Any], order: int | None = None, **kwargs
    ) -> Dict[HealpixPixel, Delayed]:
        """Perform a function on each partition of the catalog, returning a dict of values for each pixel.

        Args:
            func (Callable[[pd.DataFrame, HealpixPixel], Any]): A function that takes a pandas
                DataFrame with the data in a partition, the HealpixPixel of the partition, and any other
                keyword arguments and returns an aggregated value
            order (int | None): The HEALPix order to compute the skymap at. If None (default), will compute
                for each partition in the catalog at their own orders
            **kwargs: Arguments to pass to the function

        Returns:
            A dict of Delayed values, one for the function applied to each partition of the catalog
        """

        partitions = self.to_delayed()
        if order is None:
            results = {
                pixel: delayed(func)(partitions[index], pixel, **kwargs)
                for pixel, index in self._ddf_pixel_map.items()
            }
        else:
            if order < self.hc_structure.pixel_tree.get_max_depth():
                raise ValueError(
                    f"order must be greater than or equal to max order in catalog "
                    f"({self.hc_structure.pixel_tree.get_max_depth()})"
                )
            results = {
                pixel: perform_inner_skymap(partitions[index], func, pixel, order, **kwargs)
                for pixel, index in self._ddf_pixel_map.items()
            }
        return results

    def skymap_histogram(
        self,
        func: Callable[[pd.DataFrame, HealpixPixel], Any],
        order: int | None = None,
        default_value: Any = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Get a histogram with the result of a given function applied to the points in each HEALPix pixel of
            a given order

        Args:
            func (Callable[[pd.DataFrame], HealpixPixel, Any]): A function that takes a pandas DataFrame and
                the HealpixPixel the partition is from and returns a value
            order (int | None): The HEALPix order to compute the skymap at. If None (default), will compute
                for each partition in the catalog at their own orders
            default_value (Any): The value to use at pixels that aren't covered by the catalog (default 0)
            **kwargs: Arguments to pass to the given function

        Returns:
            A 1-dimensional numpy array where each index i is equal to the value of the function applied to
            the points within the HEALPix pixel with pixel number i in NESTED ordering at a specified order.
            If no order is supplied, the order of the resulting histogram will be the highest order partition
            in the catalog, and the function will be applied to the partitions of the catalog with the result
            copied to all pixels if the catalog partition is at a lower order than the histogram order.
        """

        smdata = self.skymap_data(func, order, **kwargs)
        pixels = list(smdata.keys())
        results = dask.compute(*[smdata[pixel] for pixel in pixels])
        result_dict = {pixels[i]: results[i] for i in range(len(pixels))}

        return compute_skymap(result_dict, order, default_value)

    def skymap(
        self,
        func: Callable[[pd.DataFrame, HealpixPixel], Any],
        order: int | None = None,
        default_value: Any = hp.pixelfunc.UNSEEN,
        projection="moll",
        plotting_args: Dict | None = None,
        **kwargs,
    ):
        """Plot a skymap of an aggregate function applied over each partition

        Args:
            func (Callable[[pd.DataFrame], HealpixPixel, Any]): A function that takes a pandas DataFrame and
                the HealpixPixel the partition is from and returns a value
            order (int | None): The HEALPix order to compute the skymap at. If None (default), will compute
                for each partition in the catalog at their own orders
            default_value (Any): The value to use at pixels that aren't covered by the catalog (default 0)
            projection (str): The map projection to use. Valid values include:
                - moll - Molleweide projection (default)
                - gnom - Gnomonic projection
                - cart - Cartesian projection
                - orth - Orthographic projection
            plotting_args (dict): A dictionary of additional arguments to pass to the plotting function
            **kwargs: Arguments to pass to the given function
        """

        img = self.skymap_histogram(func, order, default_value, **kwargs)
        projection_method = get_projection_method(projection)
        if plotting_args is None:
            plotting_args = {}
        projection_method(img, nest=True, **plotting_args)

    def plot_pixels(self, projection: str = "moll", **kwargs):
        """Create a visual map of the pixel density of the catalog.

        Args:
            projection (str) The map projection to use. Valid values include:
                - moll - Molleweide projection (default)
                - gnom - Gnomonic projection
                - cart - Cartesian projection
                - orth - Orthographic projection
            kwargs (dict): additional keyword arguments to pass to plotting call.
        """
        plot_pixel_list(self.get_healpix_pixels(), projection, **kwargs)

    def to_hipscat(
        self,
        base_catalog_path: str,
        catalog_name: str | None = None,
        overwrite: bool = False,
        storage_options: dict | None = None,
        **kwargs,
    ):
        """Saves the catalog to disk in HiPSCat format

        Args:
            base_catalog_path (str): Location where catalog is saved to
            catalog_name (str): The name of the catalog to be saved
            overwrite (bool): If True existing catalog is overwritten
            storage_options (dict): Dictionary that contains abstract filesystem credentials
            **kwargs: Arguments to pass to the parquet write operations
        """
        io.to_hipscat(self, base_catalog_path, catalog_name, overwrite, storage_options, **kwargs)
