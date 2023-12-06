from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple, Type, Union, cast

import dask.dataframe as dd
import hipscat as hc
import numpy as np
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from lsdb import io
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.cone_search import cone_filter
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data
from lsdb.dask.join_catalog_data import join_catalog_data_on
from lsdb.types import DaskDFPixelMap


# pylint: disable=R0903, W0212
class Catalog(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hipscat.Catalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: hc.catalog.Catalog

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

    def get_healpix_pixels(self) -> List[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog

        Returns:
            List of all Healpix pixels in the catalog
        """
        return self.hc_structure.get_healpix_pixels()

    def get_ordered_healpix_pixels(self) -> List[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog

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

    @property
    def name(self):
        """The name of the catalog"""
        return self.hc_structure.catalog_name

    def query(self, expr: str) -> Catalog:
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
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)

    def assign(self, **kwargs) -> Catalog:
        """Assigns new columns to a catalog

        Args:
            **kwargs: Arguments to pass to the assign method. This dictionary
                should contain the column names as keys and either a
                function or a 1-D Dask array as their corresponding value.

        Returns:
            The catalog containing both the old columns and the newly created columns
        """
        ddf = self._ddf.assign(**kwargs)
        return Catalog(ddf, self._ddf_pixel_map, self.hc_structure)

    def crossmatch(
        self,
        other: Catalog,
        suffixes: Tuple[str, str] | None = None,
        algorithm: Type[AbstractCrossmatchAlgorithm]
        | BuiltInCrossmatchAlgorithm = BuiltInCrossmatchAlgorithm.KD_TREE,
        output_catalog_name: str | None = None,
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
            algorithm (BuiltInCrossmatchAlgorithm | Type[AbstractCrossmatchAlgorithm]): The
                algorithm to use to perform the crossmatch. Can be either a string to specify one of
                the built-in cross-matching methods, or a custom method defined by subclassing
                AbstractCrossmatchAlgorithm.

                Built-in methods:
                    -`kd_tree`: find the k-nearest neighbors using a kd_tree

                Custom function:
                    To specify a custom function, write a class that subclasses the
                    `AbstractCrossmatchAlgorithm` class, and overwrite the `crossmatch` function.

                    The function should be able to perform a crossmatch on two pandas DataFrames
                    from a HEALPix pixel from each catalog. It should return a dataframe with the
                    combined set of columns from the input dataframes with the appropriate suffixes,
                    and a column with the name {AbstractCrossmatchAlgorithm.DISTANCE_COLUMN_NAME}
                    with the distance between the points.

                    The class will have been initialized with the following parameters, which the
                    crossmatch function should use:

                        - left: pd.DataFrame,
                        - right: pd.DataFrame,
                        - left_order: int,
                        - left_pixel: int,
                        - right_order: int,
                        - right_pixel: int,
                        - left_metadata: hc.catalog.Catalog,
                        - right_metadata: hc.catalog.Catalog,
                        - suffixes: Tuple[str, str]

                    You may add any additional keyword argument parameters to the crossmatch
                    function definition, and the user will be able to pass them in as kwargs in the
                    `Catalog.crossmatch` method.

            output_catalog_name (str): The name of the resulting catalog.
                Default: {left_name}_x_{right_name}

        Returns:
            A Catalog with the data from the left and right catalogs merged with one row for each
            pair of neighbors found from cross-matching.

            The resulting table contains all columns from the left and right catalogs with their
            respective suffixes, and a column with the name
            {AbstractCrossmatchAlgorithm.DISTANCE_COLUMN_NAME} with the great circle separation
            between the points.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")
        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")
        if output_catalog_name is None:
            output_catalog_name = f"{self.name}_x_{other.name}"
        ddf, ddf_map, alignment = crossmatch_catalog_data(
            self, other, suffixes, algorithm=algorithm, **kwargs
        )
        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            catalog_name=output_catalog_name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    @staticmethod
    def _check_ra_dec_values_valid(ra: float, dec: float):
        if ra < -180 or ra > 180:
            raise ValueError("ra must be between -180 and 180")
        if dec > 90 or dec < -90:
            raise ValueError("dec must be between -90 and 90")

    def cone_search(self, ra: float, dec: float, radius: float):
        """Perform a cone search to filter the catalog

        Filters to points within radius great circle distance to the point specified by ra and dec in degrees.
        Filters partitions in the catalog to those that have some overlap with the cone.

        Args:
            ra (float): Right Ascension of the center of the cone in degrees
            dec (float): Declination of the center of the cone in degrees
            radius (float): Radius of the cone in degrees

        Returns:
            A new Catalog containing the points filtered to those within the cone, and the partitions that
            overlap the cone.
        """
        if radius < 0:
            raise ValueError("Cone radius must be non negative")
        self._check_ra_dec_values_valid(ra, dec)
        filtered_hc_structure = self.hc_structure.filter_by_cone(ra, dec, radius)
        pixels_in_cone = filtered_hc_structure.get_healpix_pixels()
        partitions = self._ddf.to_delayed()
        partitions_in_cone = [partitions[self._ddf_pixel_map[pixel]] for pixel in pixels_in_cone]
        filtered_partitions = [
            cone_filter(partition, ra, dec, radius, self.hc_structure) for partition in partitions_in_cone
        ]
        cone_search_ddf = dd.from_delayed(filtered_partitions, meta=self._ddf._meta)
        cone_search_ddf = cast(dd.DataFrame, cone_search_ddf)
        ddf_partition_map = {pixel: i for i, pixel in enumerate(pixels_in_cone)}
        return Catalog(cone_search_ddf, ddf_partition_map, filtered_hc_structure)

    def merge(
        self,
        other: Catalog,
        how: str = "inner",
        on: str | List | None = None,
        left_on: str | List | None = None,
        right_on: str | List | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Tuple[str, str] | None = None,
    ) -> dd.DataFrame:
        """Performs the merge of two catalog Dataframes

        More information about pandas merge is available
        `here <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html>`__.

        Args:
            other (Catalog): The right catalog to merge with.
            how (str): How to handle the merge of the two catalogs.
                One of {'left', 'right', 'outer', 'inner'}, defaults to 'inner'.
            on (str | List): Column or index names to join on. Defaults to the
                intersection of columns in both Dataframes if on is None and not
                merging on indexes.
            left_on (str | List): Column to join on the left Dataframe. Lists are
                supported if their length is one.
            right_on (str | List): Column to join on the right Dataframe. Lists are
                supported if their length is one.
            left_index (bool): Use the index of the left Dataframe as the join key.
                Defaults to False.
            right_index (bool): Use the index of the right Dataframe as the join key.
                Defaults to False.
            suffixes (Tuple[str, str]): A pair of suffixes to be appended to the
                end of each column name when they are joined. Defaults to using the
                name of the catalog for the suffix.

        Returns:
            A new Dask Dataframe containing the data points that result from the merge
            of the two catalogs.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")
        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")
        return self._ddf.merge(
            other._ddf,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
        )

    def to_hipscat(
        self,
        base_catalog_path: str,
        catalog_name: Union[str, None] = None,
        storage_options: Union[Dict[Any, Any], None] = None,
        **kwargs,
    ):
        """Saves the catalog to disk in HiPSCat format

        Args:
            base_catalog_path (str): Location where catalog is saved to
            catalog_name (str): The name of the catalog to be saved
            storage_options (dict): Dictionary that contains abstract filesystem credentials
            **kwargs: Arguments to pass to the parquet write operations
        """
        io.to_hipscat(self, base_catalog_path, catalog_name, storage_options, **kwargs)

    def join(
        self,
        other: Catalog,
        left_on: str,
        right_on: str,
        suffixes: Tuple[str, str] | None = None,
        output_catalog_name: str | None = None,
    ) -> Catalog:
        """Perform a spatial join to another catalog

        Joins two catalogs together on a shared column value, merging rows where they match. The operation
        only joins data from matching partitions, and does not join rows that have a matching column value but
        are in separate partitions in the sky. For a more general join, see the `merge` function.

        Args:
            other (Catalog): the right catalog to join to
            left_on (str): the name of the column in the left catalog to join on
            right_on (str): the name of the column in the right catalog to join on
            suffixes (Tuple[str,str]): suffixes to apply to the columns of each table
            output_catalog_name (str): The name of the resulting catalog to be stored in metadata

        Returns:
            A new catalog with the columns from each of the input catalogs with their respective suffixes
            added, and the rows merged on the specified columns.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")

        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")

        if left_on not in self._ddf.columns:
            raise ValueError("left_on must be a column in the left catalog")

        if right_on not in other._ddf.columns:
            raise ValueError("right_on must be a column in the right catalog")

        ddf, ddf_map, alignment = join_catalog_data_on(self, other, left_on, right_on, suffixes=suffixes)

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = dataclasses.replace(
            self.hc_structure.catalog_info,
            catalog_name=output_catalog_name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)
