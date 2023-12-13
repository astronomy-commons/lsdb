from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Tuple

import dask.dataframe as dd
import hipscat as hc
import numpy as np
import pandas as pd
from dask import delayed
from hipscat.catalog import CatalogType
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel, generate_histogram
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN, compute_hipscat_id, healpix_to_hipscat_id

from lsdb.catalog.catalog import Catalog
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.types import DaskDFPixelMap, HealpixInfo

pd.options.mode.chained_assignment = None  # default='warn'


class DataframeCatalogLoader:
    """Creates a HiPSCat formatted Catalog from a Pandas Dataframe"""

    DEFAULT_THRESHOLD = 100_000

    def __init__(
        self,
        dataframe: pd.DataFrame,
        lowest_order: int = 0,
        highest_order: int = 5,
        partition_size: int | None = None,
        threshold: int | None = None,
        **kwargs,
    ) -> None:
        """Initializes a DataframeCatalogLoader

        Args:
            dataframe (pd.Dataframe): Catalog Pandas Dataframe
            lowest_order (int): The lowest partition order
            highest_order (int): The highest partition order
            partition_size (int): The desired partition size, in number of rows
            threshold (int): The maximum number of data points per pixel
            **kwargs: Arguments to pass to the creation of the catalog info
        """
        self.dataframe = dataframe
        self.lowest_order = lowest_order
        self.highest_order = highest_order
        self.threshold = self._calculate_threshold(partition_size, threshold)
        self.catalog_info = self._create_catalog_info(**kwargs)

    def _calculate_threshold(self, partition_size: int | None = None, threshold: int | None = None) -> int:
        """Calculates the number of pixels per HEALPix pixel (threshold) for the
        desired partition size.

        Args:
            partition_size (int): The desired partition size, in number of rows
            threshold (int): The maximum number of data points per pixel

        Returns:
            The HEALPix pixel threshold
        """
        if threshold is not None and partition_size is not None:
            raise ValueError("Specify only one: threshold or partition_size")
        if threshold is None:
            if partition_size is not None:
                # Round the number of partitions to the next integer, otherwise the
                # number of pixels per partition may exceed the threshold
                num_partitions = math.ceil(len(self.dataframe) / partition_size)
                threshold = len(self.dataframe.index) // num_partitions
            else:
                threshold = DataframeCatalogLoader.DEFAULT_THRESHOLD
        return threshold

    @staticmethod
    def _create_catalog_info(**kwargs) -> CatalogInfo:
        """Creates the catalog info object

        Args:
            **kwargs: Arguments to pass to the creation of the catalog info

        Returns:
            The catalog info object
        """
        valid_catalog_types = [CatalogType.OBJECT, CatalogType.SOURCE]
        catalog_info = CatalogInfo(**kwargs)
        if catalog_info.catalog_type not in valid_catalog_types:
            raise ValueError("Catalog must be of type OBJECT or SOURCE")
        return catalog_info

    def load_catalog(self) -> Catalog:
        """Load a catalog from a Pandas Dataframe, in CSV format

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        self._set_hipscat_index()
        pixel_map = self._compute_pixel_map()
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map(pixel_map)
        self.catalog_info = dataclasses.replace(self.catalog_info, total_rows=total_rows)
        healpix_pixels = list(pixel_map.keys())
        hc_structure = hc.catalog.Catalog(self.catalog_info, healpix_pixels)
        return Catalog(ddf, ddf_pixel_map, hc_structure)

    def _set_hipscat_index(self):
        """Generates the hipscat indices for each data point and assigns
        the hipscat index column as the Dataframe index."""
        self.dataframe[HIPSCAT_ID_COLUMN] = compute_hipscat_id(
            ra_values=self.dataframe[self.catalog_info.ra_column].values,
            dec_values=self.dataframe[self.catalog_info.dec_column].values,
        )
        self.dataframe.set_index(HIPSCAT_ID_COLUMN, inplace=True)

    def _compute_pixel_map(self) -> Dict[HealpixPixel, HealpixInfo]:
        """Compute object histogram and generate the sorted mapping between
        HEALPix pixels and the respective original pixel information. The
        pixels are sorted by ascending hipscat_id.

        Returns:
            A dictionary mapping each HEALPix pixel to the respective
            information tuple. The first value of the tuple is the number
            of objects in the HEALPix pixel, the second is the list of pixels.
        """
        raw_histogram = generate_histogram(
            self.dataframe,
            highest_order=self.highest_order,
            ra_column=self.catalog_info.ra_column,
            dec_column=self.catalog_info.dec_column,
        )
        pixel_map = hc.pixel_math.compute_pixel_map(
            raw_histogram,
            highest_order=self.highest_order,
            lowest_order=self.lowest_order,
            threshold=self.threshold,
        )
        pixels = list(pixel_map.keys())
        ordered_pixels = np.array(pixels)[get_pixel_argsort(pixels)]
        return {pixel: pixel_map[pixel] for pixel in ordered_pixels}

    def _generate_dask_df_and_map(
        self, pixel_map: Dict[HealpixPixel, HealpixInfo]
    ) -> Tuple[dd.DataFrame, DaskDFPixelMap, int]:
        """Load Dask DataFrame from HEALPix pixel Dataframes and
        generate a mapping of HEALPix pixels to HEALPix Dataframes

        Args:
            pixel_map (Dict[HealpixPixel, HealpixInfo]): The mapping between
                catalog HEALPix pixels and respective data information.

        Returns:
            Tuple containing the Dask Dataframe, the mapping of HEALPix pixels
            to the respective Pandas Dataframes and the total number of rows.
        """
        # Dataframes for each destination HEALPix pixel
        pixel_dfs: List[pd.DataFrame] = []

        # Mapping HEALPix pixels to the respective Dataframe indices
        ddf_pixel_map: Dict[HealpixPixel, int] = {}

        # Dask Dataframe divisions
        divisions = get_pixels_divisions(list(pixel_map.keys()))

        for hp_pixel_index, hp_pixel_info in enumerate(pixel_map.items()):
            hp_pixel, (_, pixels) = hp_pixel_info
            # Store HEALPix pixel in map
            ddf_pixel_map[hp_pixel] = hp_pixel_index
            # Obtain Dataframe for current HEALPix pixel
            pixel_dfs.append(self._get_dataframe_for_healpix(hp_pixel, pixels))

        # Generate Dask Dataframe with original schema
        schema = pixel_dfs[0].iloc[:0, :].copy()
        ddf, total_rows = self._generate_dask_dataframe(pixel_dfs, schema, divisions)
        return ddf, ddf_pixel_map, total_rows

    @staticmethod
    def _generate_dask_dataframe(
        pixel_dfs: List[pd.DataFrame], schema: pd.DataFrame, divisions: Tuple[int, ...] | None
    ) -> Tuple[dd.DataFrame, int]:
        """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

        Args:
            pixel_dfs (List[pd.DataFrame]): The list of HEALPix pixel Dataframes
            schema (pd.Dataframe): The original Dataframe schema
            divisions (Tuple[int, ...]): The partitions divisions

        Returns:
            The catalog's Dask Dataframe and its total number of rows.
        """
        delayed_dfs = [delayed(df) for df in pixel_dfs]
        ddf = dd.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
        return ddf if isinstance(ddf, dd.DataFrame) else ddf.to_frame(), len(ddf)

    def _get_dataframe_for_healpix(self, hp_pixel: HealpixPixel, pixels: List[int]) -> pd.DataFrame:
        """Computes the Pandas Dataframe containing the data points
        for a certain HEALPix pixel.

        Using NESTED ordering scheme, the provided list is a sequence of contiguous
        pixel numbers, in ascending order, inside the HEALPix pixel. Therefore, the
        corresponding points in the Dataframe will be located between the hipscat
        index of the lowest numbered pixel (left_bound) and the hipscat index of the
        highest numbered pixel (right_bound).

        Args:
            hp_pixel (HealpixPixel): The HEALPix pixel to generate the Dataframe for
            pixels (List[int]): The indices of the pixels inside the HEALPix pixel

        Returns:
            The Pandas Dataframe containing the data points for the HEALPix pixel.
        """
        left_bound = healpix_to_hipscat_id(self.highest_order, pixels[0])
        right_bound = healpix_to_hipscat_id(self.highest_order, pixels[-1] + 1)
        pixel_df = self.dataframe.loc[
            (self.dataframe.index >= left_bound) & (self.dataframe.index < right_bound)
        ]
        return self._append_partition_information_to_dataframe(pixel_df, hp_pixel)

    def _append_partition_information_to_dataframe(
        self, dataframe: pd.DataFrame, pixel: HealpixPixel
    ) -> pd.DataFrame:
        """Appends partitioning information to a HEALPix dataframe

        Args:
            dataframe (pd.Dataframe): A HEALPix's pandas dataframe
            pixel (HealpixPixel): The HEALPix pixel for the current partition

        Returns:
            The dataframe for a HEALPix, with data points and respective partition information.
        """
        ordered_columns = ["Norder", "Dir", "Npix"]
        # Generate partition information
        dataframe["Norder"] = pixel.order
        dataframe["Npix"] = pixel.pixel
        dataframe["Dir"] = pixel.dir
        # Force new column types to int
        dataframe[ordered_columns] = dataframe[ordered_columns].astype(int)
        # Reorder the columns to match full path
        return dataframe[[col for col in dataframe.columns if col not in ordered_columns] + ordered_columns]
