from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Dict, List, Tuple

import astropy.units as u
import dask.dataframe as dd
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat.catalog import CatalogType
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.pixel_math import HealpixPixel, generate_histogram
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort
from hipscat.pixel_math.hipscat_id import HIPSCAT_ID_COLUMN, compute_hipscat_id, healpix_to_hipscat_id
from mocpy import MOC

from lsdb.catalog.catalog import Catalog
from lsdb.loaders.dataframe.from_dataframe_utils import (
    _append_partition_information_to_dataframe,
    _generate_dask_dataframe,
)
from lsdb.types import DaskDFPixelMap

pd.options.mode.chained_assignment = None  # default='warn'


class DataframeCatalogLoader:
    """Creates a HiPSCat formatted Catalog from a Pandas Dataframe"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        lowest_order: int = 0,
        highest_order: int = 7,
        drop_empty_siblings: bool = False,
        partition_size: int | None = None,
        threshold: int | None = None,
        should_generate_moc: bool = True,
        moc_max_order: int = 10,
        use_pyarrow_types: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a DataframeCatalogLoader

        Args:
            dataframe (pd.Dataframe): Catalog Pandas Dataframe.
            lowest_order (int): The lowest partition order. Defaults to 3.
            highest_order (int): The highest partition order. Defaults to 7.
            drop_empty_siblings (bool): When determining final partitionining,
                if 3 of 4 pixels are empty, keep only the non-empty pixel
            partition_size (int): The desired partition size, in number of bytes in-memory.
            threshold (int): The maximum number of data points per pixel.
            should_generate_moc (bool): should we generate a MOC (multi-order coverage map)
                of the data. can improve performance when joining/crossmatching to
                other hipscatted datasets.
            moc_max_order (int): if generating a MOC, what to use as the max order. Defaults to 10.
            use_pyarrow_types (bool): If True, the data is backed by pyarrow, otherwise we keep the
                original data types. Defaults to True.
            **kwargs: Arguments to pass to the creation of the catalog info.
        """
        self.dataframe = dataframe
        self.lowest_order = lowest_order
        self.highest_order = highest_order
        self.drop_empty_siblings = drop_empty_siblings
        self.threshold = self._calculate_threshold(partition_size, threshold)
        self.catalog_info = self._create_catalog_info(**kwargs)
        self.should_generate_moc = should_generate_moc
        self.moc_max_order = moc_max_order
        self.use_pyarrow_types = use_pyarrow_types

    def _calculate_threshold(self, partition_size: int | None = None, threshold: int | None = None) -> int:
        """Calculates the number of pixels per HEALPix pixel (threshold) for the
        desired partition size.

        Args:
            partition_size (int): The desired partition size, in number of rows
            threshold (int): The maximum number of data points per pixel

        Returns:
            The HEALPix pixel threshold
        """
        df_total_memory = self.dataframe.memory_usage(deep=True).sum()
        if df_total_memory > (1 << 30) or len(self.dataframe) > 1_000_000:
            warnings.warn(
                "from_dataframe is not intended for large datasets. "
                "Consider using hipscat-import: https://hipscat-import.readthedocs.io/",
                RuntimeWarning,
            )
        if threshold is not None and partition_size is not None:
            raise ValueError("Specify only one: threshold or partition_size")
        if threshold is None:
            if partition_size is not None:
                # Round the number of partitions to the next integer, otherwise the
                # number of pixels per partition may exceed the threshold
                num_partitions = math.ceil(len(self.dataframe) / partition_size)
                threshold = len(self.dataframe) // num_partitions
            else:
                # Each partition in memory will be of roughly 1Gib
                partition_memory = df_total_memory / len(self.dataframe)
                threshold = math.ceil((1 << 30) / partition_memory)
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
        pixel_list = self._compute_pixel_list()
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map(pixel_list)
        self.catalog_info = dataclasses.replace(self.catalog_info, total_rows=total_rows)
        moc = self._generate_moc() if self.should_generate_moc else None
        hc_structure = hc.catalog.Catalog(self.catalog_info, pixel_list, moc=moc)
        return Catalog(ddf, ddf_pixel_map, hc_structure)

    def _set_hipscat_index(self):
        """Generates the hipscat indices for each data point and assigns
        the hipscat index column as the Dataframe index."""
        self.dataframe[HIPSCAT_ID_COLUMN] = compute_hipscat_id(
            ra_values=self.dataframe[self.catalog_info.ra_column].to_numpy(),
            dec_values=self.dataframe[self.catalog_info.dec_column].to_numpy(),
        )
        self.dataframe.set_index(HIPSCAT_ID_COLUMN, inplace=True)

    def _compute_pixel_list(self) -> List[HealpixPixel]:
        """Compute object histogram and generate the sorted list of
        HEALPix pixels. The pixels are sorted by ascending hipscat_id.

        Returns:
            List of HEALPix pixels for the final partitioning.
        """
        raw_histogram = generate_histogram(
            self.dataframe,
            highest_order=self.highest_order,
            ra_column=self.catalog_info.ra_column,
            dec_column=self.catalog_info.dec_column,
        )
        alignment = hc.pixel_math.generate_alignment(
            raw_histogram,
            highest_order=self.highest_order,
            lowest_order=self.lowest_order,
            threshold=self.threshold,
            drop_empty_siblings=self.drop_empty_siblings,
        )
        non_none_elements = alignment[alignment != np.array(None)]
        pixel_list = np.unique(non_none_elements)
        pixel_list = [HealpixPixel(order, pix) for (order, pix, count) in pixel_list if int(count) > 0]
        return list(np.array(pixel_list)[get_pixel_argsort(pixel_list)])

    def _generate_dask_df_and_map(
        self, pixel_list: List[HealpixPixel]
    ) -> Tuple[dd.DataFrame, DaskDFPixelMap, int]:
        """Load Dask DataFrame from HEALPix pixel Dataframes and
        generate a mapping of HEALPix pixels to HEALPix Dataframes

        Args:
            pixel_list (List[HealpixPixel]): final partitioning of data

        Returns:
            Tuple containing the Dask Dataframe, the mapping of HEALPix pixels
            to the respective Pandas Dataframes and the total number of rows.
        """
        # Dataframes for each destination HEALPix pixel
        pixel_dfs: List[pd.DataFrame] = []

        # Mapping HEALPix pixels to the respective Dataframe indices
        ddf_pixel_map: Dict[HealpixPixel, int] = {}

        for hp_pixel_index, hp_pixel in enumerate(pixel_list):
            # Store HEALPix pixel in map
            ddf_pixel_map[hp_pixel] = hp_pixel_index
            # Obtain Dataframe for current HEALPix pixel, using NESTED characteristics.
            left_bound = healpix_to_hipscat_id(hp_pixel.order, hp_pixel.pixel)
            right_bound = healpix_to_hipscat_id(hp_pixel.order, hp_pixel.pixel + 1)
            pixel_df = self.dataframe.loc[
                (self.dataframe.index >= left_bound) & (self.dataframe.index < right_bound)
            ]
            pixel_dfs.append(_append_partition_information_to_dataframe(pixel_df, hp_pixel))

        # Generate Dask Dataframe with the original schema and desired backend
        ddf, total_rows = _generate_dask_dataframe(pixel_dfs, pixel_list, self.use_pyarrow_types)
        return ddf, ddf_pixel_map, total_rows

    def _generate_moc(self):
        lon = self.dataframe[self.catalog_info.ra_column].to_numpy() * u.deg
        lat = self.dataframe[self.catalog_info.dec_column].to_numpy() * u.deg
        return MOC.from_lonlat(lon=lon, lat=lat, max_norder=self.moc_max_order)
