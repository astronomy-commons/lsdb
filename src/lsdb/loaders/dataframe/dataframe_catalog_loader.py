from __future__ import annotations

import math
import re
import warnings

import astropy.units as u
import hats as hc
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from hats.catalog import CatalogType, TableProperties
from hats.pixel_math import HealpixPixel, generate_histogram
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import (
    SPATIAL_INDEX_COLUMN,
    compute_spatial_index,
    healpix_to_spatial_index,
)
from mocpy import MOC

import lsdb.nested as nd
from lsdb.catalog.catalog import Catalog
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.dataframe.from_dataframe_utils import _generate_dask_dataframe, _has_named_index
from lsdb.types import DaskDFPixelMap

pd.options.mode.chained_assignment = None  # default='warn'


class DataframeCatalogLoader:
    """Creates a HATS formatted Catalog from a Pandas Dataframe"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        ra_column: str | None = None,
        dec_column: str | None = None,
        lowest_order: int = 0,
        highest_order: int = 7,
        drop_empty_siblings: bool = False,
        partition_size: int | None = None,
        threshold: int | None = None,
        should_generate_moc: bool = True,
        moc_max_order: int = 10,
        use_pyarrow_types: bool = True,
        schema: pa.Schema | None = None,
        **kwargs,
    ) -> None:
        """Initializes a DataframeCatalogLoader

        Args:
            dataframe (pd.Dataframe): Catalog Pandas Dataframe.
            ra_column (str): The name of the right ascension column. By default,
                case-insensitive versions of 'ra' are detected.
            dec_column (str): The name of the declination column. By default,
                case-insensitive versions of 'dec' are detected.
            lowest_order (int): The lowest partition order. Defaults to 3.
            highest_order (int): The highest partition order. Defaults to 7.
            drop_empty_siblings (bool): When determining final partitionining,
                if 3 of 4 pixels are empty, keep only the non-empty pixel
            partition_size (int): The desired partition size, in number of rows.
            threshold (int): The maximum number of data points per pixel.
            should_generate_moc (bool): should we generate a MOC (multi-order coverage map)
                of the data. can improve performance when joining/crossmatching to
                other hats-sharded datasets.
            moc_max_order (int): if generating a MOC, what to use as the max order. Defaults to 10.
            use_pyarrow_types (bool): If True, the data is backed by pyarrow, otherwise we keep the
                original data types. Defaults to True.
            schema (pa.Schema): the arrow schema to create the catalog with. If None, the schema is
                automatically inferred from the provided DataFrame using `pa.Schema.from_pandas`.
            **kwargs: Arguments to pass to the creation of the catalog info.
        """
        self.dataframe = npd.NestedFrame(dataframe)
        self.lowest_order = lowest_order
        self.highest_order = highest_order
        self.drop_empty_siblings = drop_empty_siblings
        self.threshold = self._calculate_threshold(partition_size, threshold)

        if ra_column is None:
            ra_column = self._find_column("ra")
        if dec_column is None:
            dec_column = self._find_column("dec")

        self.catalog_info = self._create_catalog_info(
            ra_column=ra_column,
            dec_column=dec_column,
            total_rows=len(self.dataframe),
            hats_max_rows=self.threshold,
            **kwargs,
        )
        self.should_generate_moc = should_generate_moc
        self.moc_max_order = moc_max_order
        self.use_pyarrow_types = use_pyarrow_types
        self.schema = schema

    def _find_column(self, search_term: str) -> str:
        """Finds the column in the data frame matching the search term.

        The search is case-insensitive and unambiguous. An error is raised
        if there are zero or multiple matches.

        Args:
            search_term (str): The column name to search for.

        Returns:
            The column name.
        """
        matches = [
            c for c in self.dataframe.columns if re.fullmatch(re.escape(search_term), c, re.IGNORECASE)
        ]
        n_matches = len(matches)
        if n_matches == 0:
            raise ValueError(f"No column found for {search_term}")
        if n_matches > 1:
            raise ValueError(f"Found {n_matches} possible columns for {search_term}")
        return matches[0]

    def _calculate_threshold(self, partition_size: int | None = None, threshold: int | None = None) -> int:
        """Calculates the number of pixels per HEALPix pixel (threshold) for the
        desired partition size.

        Args:
            partition_size (int): The desired partition size, in number of rows
            threshold (int): The maximum number of data points per pixel

        Returns:
            The HEALPix pixel threshold
        """
        self.df_total_memory = self.dataframe.memory_usage(deep=True).sum()
        if self.df_total_memory > (1 << 30) or len(self.dataframe) > 1_000_000:
            warnings.warn(
                "from_dataframe is not intended for large datasets. "
                "Consider using hats-import: https://hats-import.readthedocs.io/",
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
                partition_memory = self.df_total_memory / len(self.dataframe)
                threshold = math.ceil((1 << 30) / partition_memory)
        return threshold

    def _create_catalog_info(
        self,
        catalog_name: str = "from_lsdb_dataframe",
        ra_column: str = "ra",
        dec_column: str = "dec",
        catalog_type: CatalogType = CatalogType.OBJECT,
        **kwargs,
    ) -> TableProperties:
        """Creates the catalog info object

        Args:
            catalog_name: it is recommended to provide a new name for your catalog
            ra_column: column to find right ascension coordinate
            dec_column: column to find declination coordinate
            catalog_type: type of table being created (e.g. OBJECT, SOURCE, MAP)
            **kwargs: Arguments to pass to the creation of the catalog info

        Returns:
            The catalog info object
        """
        if kwargs is None:
            kwargs = {}
        kwargs = kwargs | Dataset.new_provenance_properties(hats_estsize=self.df_total_memory)
        if catalog_type and catalog_type not in (CatalogType.OBJECT, CatalogType.SOURCE, CatalogType.MAP):
            raise ValueError(f"Cannot create {catalog_type} type catalog via from_dataframe.")
        return TableProperties(
            catalog_name=catalog_name,
            ra_column=ra_column,
            dec_column=dec_column,
            catalog_type=catalog_type,
            **kwargs,
        )

    def load_catalog(self) -> Catalog:
        """Load a catalog from a Pandas Dataframe

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        self._set_spatial_index()
        pixel_list = self._compute_pixel_list()
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map(pixel_list)
        self.catalog_info.total_rows = total_rows
        moc = self._generate_moc() if self.should_generate_moc else None
        schema = self.schema if self.schema is not None else get_arrow_schema(ddf)
        hc_structure = hc.catalog.Catalog(self.catalog_info, pixel_list, moc=moc, schema=schema)

        # Recover NestedDtype (https://github.com/astronomy-commons/lsdb/issues/730)
        if len(self.dataframe.nested_columns) > 0:
            ddf = ddf.astype({col: self.dataframe[col].dtype for col in self.dataframe.nested_columns})

        return Catalog(ddf, ddf_pixel_map, hc_structure)

    def _set_spatial_index(self):
        """Generates the spatial indices for each data point and assigns
        the spatial index column as the Dataframe index."""
        if _has_named_index(self.dataframe):
            self.dataframe.reset_index(inplace=True)
        self.dataframe[SPATIAL_INDEX_COLUMN] = compute_spatial_index(
            ra_values=self.dataframe[self.catalog_info.ra_column].to_numpy(),
            dec_values=self.dataframe[self.catalog_info.dec_column].to_numpy(),
        )
        self.dataframe.set_index(SPATIAL_INDEX_COLUMN, inplace=True)

    def _compute_pixel_list(self) -> list[HealpixPixel]:
        """Compute object histogram and generate the sorted list of
        HEALPix pixels. The pixels are sorted by ascending spatial index.

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
        pixel_list = list({HealpixPixel(tup[0], tup[1]) for tup in alignment if not tup is None})
        return list(np.array(pixel_list)[get_pixel_argsort(pixel_list)])

    def _generate_dask_df_and_map(
        self, pixel_list: list[HealpixPixel]
    ) -> tuple[nd.NestedFrame, DaskDFPixelMap, int]:
        """Load Dask DataFrame from HEALPix pixel Dataframes and
        generate a mapping of HEALPix pixels to HEALPix Dataframes

        Args:
            pixel_list (List[HealpixPixel]): final partitioning of data

        Returns:
            Tuple containing the Dask Dataframe, the mapping of HEALPix pixels
            to the respective Pandas Dataframes and the total number of rows.
        """
        # Dataframes for each destination HEALPix pixel
        pixel_dfs: list[npd.NestedFrame] = []

        # Mapping HEALPix pixels to the respective Dataframe indices
        ddf_pixel_map: dict[HealpixPixel, int] = {}

        for hp_pixel_index, hp_pixel in enumerate(pixel_list):
            # Store HEALPix pixel in map
            ddf_pixel_map[hp_pixel] = hp_pixel_index
            # Obtain Dataframe for current HEALPix pixel, using NESTED characteristics.
            left_bound = healpix_to_spatial_index(hp_pixel.order, hp_pixel.pixel)
            right_bound = healpix_to_spatial_index(hp_pixel.order, hp_pixel.pixel + 1)
            pixel_df = self.dataframe.loc[
                (self.dataframe.index >= left_bound) & (self.dataframe.index < right_bound)
            ]
            pixel_dfs.append(pixel_df)

        # Generate Dask Dataframe with the original schema and desired backend
        ddf, total_rows = _generate_dask_dataframe(pixel_dfs, pixel_list, self.use_pyarrow_types)
        return ddf, ddf_pixel_map, total_rows

    def _generate_moc(self):
        lon = self.dataframe[self.catalog_info.ra_column].to_numpy() * u.deg
        lat = self.dataframe[self.catalog_info.dec_column].to_numpy() * u.deg
        return MOC.from_lonlat(lon=lon, lat=lat, max_norder=self.moc_max_order)
