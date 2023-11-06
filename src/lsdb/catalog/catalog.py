from __future__ import annotations

import dataclasses
import os
from importlib.metadata import version
from typing import Any, Dict, List, Tuple, Type, Union, cast

import dask.dataframe as dd
import hipscat as hc
import numpy as np
from hipscat.pixel_math import HealpixPixel
from typing_extensions import TypeAlias

from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.cone_search import cone_filter
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data
from lsdb.io import file_io
from lsdb.io.file_io import write_provenance_info

DaskDFPixelMap = Dict[HealpixPixel, int]

# Compute pixel map returns a tuple. The first element is
# the number of data points within the HEALPix pixel, the
# second element is the list of pixels it contains.
HealpixInfo: TypeAlias = Tuple[int, List[int]]


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

    def get_healpix_highest_order(self) -> int:
        """Returns the HEALPix pixels highest order

        Returns:
            An integer representing the healpix pixels highest order.
        """
        pixel_orders = [pixel.order for pixel in self.get_healpix_pixels()]
        return np.max(pixel_orders)

    def get_histogram(self, partition_map: Dict[HealpixPixel, int]):
        """Creates the HEALPix pixel histogram

        Args:
            partition_map (Dict[HealpixPixel, int]): The mapping between each
                HEALPix pixel and the respective number of points

        Returns:
            A one-dimensional numpy array of long integers where the value at each
            index corresponds to the number of objects found at the healpix pixel.
        """
        highest_order = self.get_healpix_highest_order()
        histogram = hc.pixel_math.empty_histogram(highest_order)
        for i, (_, num_points) in enumerate(partition_map.items()):
            histogram[i] = num_points
        return histogram

    def get_provenance_info(self) -> dict:
        """Fill all known information in a dictionary for provenance tracking.

        Returns:
            dictionary with all argument_name -> argument_value as key -> value pairs.
        """
        catalog_info = self.hc_structure.catalog_info

        runtime_args = {
            "catalog_name": self.hc_structure.catalog_name,
            "output_path": self.hc_structure.catalog_path,
            "output_catalog_name": self.hc_structure.catalog_name,
            # "tmp_dir": str(self.tmp_dir),
            # "overwrite": self.overwrite,
            # "dask_tmp": str(self.dask_tmp),
            # "dask_n_workers": self.dask_n_workers,
            # "dask_threads_per_worker": self.dask_threads_per_worker,
            "catalog_path": self.hc_structure.catalog_path,
            # "tmp_path": str(self.tmp_path),
        }
        additional_args = {
            "catalog_name": self.hc_structure.catalog_name,
            "epoch": catalog_info.epoch,
            "catalog_type": catalog_info.catalog_type,
            # "input_path": str(self.input_path),
            # "input_paths": self.input_paths,
            # "input_format": self.input_format,
            # "input_file_list": self.input_file_list,
            "ra_column": catalog_info.ra_column,
            "dec_column": catalog_info.dec_column,
            "highest_healpix_order": int(self.get_healpix_highest_order()),
            # "pixel_threshold": self.pixel_threshold,
            # "mapping_healpix_order": self.mapping_healpix_order,
            # "debug_stats_only": self.debug_stats_only,
            # "file_reader_info": self.file_reader.provenance_info() if self.file_reader is not None else {},
        }
        runtime_args.update(additional_args)
        provenance_info = {
            "tool_name": "lsdb",
            "version": version("lsdb"),
            "runtime_args": runtime_args,
        }
        return provenance_info

    def get_pixel_map(self, ddf_points_map: Dict[HealpixPixel, int]) -> Dict[HealpixPixel, HealpixInfo]:
        """Creates the partition info dictionary

        Args:
            ddf_points_map (Dict[HealpixPix,int]): Dictionary mapping each HealpixPixel
                to the respective number of points inside its partition

        Returns:
            A partition info dictionary, where the keys are the HEALPix pixels and
            the values are pairs where the first element is the number of points
            inside the pixel, and the second is the list of destination pixel numbers.
        """
        return {pixel: (length, [pixel.pixel]) for pixel, length in ddf_points_map.items()}

    @property
    def name(self):
        """The name of the catalog"""
        return self.hc_structure.catalog_name

    def update_catalog_info(self, **kwargs):
        """Updates catalog information

        Args:
            **kwargs: Parameters to update in catalog info
        """
        catalog_info = dataclasses.replace(self.hc_structure.catalog_info, **kwargs)
        self.hc_structure.catalog_info = catalog_info

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

    def to_hipscat(
            self,
            base_catalog_path: str,
            catalog_name: Union[str | None] = None,
            storage_options: Union[Dict[Any, Any], None] = None,
    ):
        """Saves the catalog to disk in HiPSCat format

        Args:
            base_catalog_path (str): Location where catalog is saved to
            catalog_name (str): The name of the catalog to be saved
            storage_options: dictionary that contains abstract filesystem credentials
        """
        os.makedirs(base_catalog_path)

        base_catalog_dir_fp = hc.io.get_file_pointer_from_path(base_catalog_path)

        # Write partition parquet files
        partition_map = {}
        for pixel, partition_index in self._ddf_pixel_map.items():
            partition = self._ddf.partitions[partition_index].compute()
            pixel_path = hc.io.paths.pixel_catalog_file(base_catalog_dir_fp, pixel.order, pixel.pixel)
            file_io.write_dataframe_to_parquet(partition, pixel_path)
            partition_map[pixel] = len(partition)

        # Write partition info
        partition_info = self.get_pixel_map(partition_map)
        hc.io.write_partition_info(base_catalog_dir_fp, partition_info, storage_options)

        # Write the catalog info (and update the catalog_name and total_rows, as needed)
        catalog_name = catalog_name if catalog_name is not None else self.hc_structure.catalog_name
        self.update_catalog_info(catalog_name=catalog_name)
        self.hc_structure.catalog_path = base_catalog_path
        hc.io.write_catalog_info(base_catalog_path, self.hc_structure.catalog_info, storage_options)

        # Write provenance info
        provenance_info = self.get_provenance_info()
        write_provenance_info(
            base_catalog_dir_fp, self.hc_structure.catalog_info, provenance_info, storage_options
        )

        # Write parquet metadata
        hc.io.write_parquet_metadata(base_catalog_path, storage_options)

        # Write fits map
        pixel_histogram = self.get_histogram(partition_map)
        hc.io.write_metadata.write_fits_map(base_catalog_path, pixel_histogram, storage_options)
