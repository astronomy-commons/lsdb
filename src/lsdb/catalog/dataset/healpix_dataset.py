from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Type

import astropy
import dask
import dask.dataframe as dd
import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from dask.dataframe.core import _repr_data_series
from deprecated import deprecated  # type: ignore
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from mocpy import MOC
from pandas._typing import Renamer
from typing_extensions import Self
from upath import UPath

import lsdb.nested as nd
from lsdb import io
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.plotting.plot_points import plot_points
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.core.search.region_search import (
    BoxSearch,
    ConeSearch,
    MOCSearch,
    OrderSearch,
    PixelSearch,
    PolygonSearch,
)
from lsdb.dask.merge_catalog_functions import concat_metas
from lsdb.dask.partition_indexer import PartitionIndexer
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.visualization.wcsaxes.frame import BaseFrame
    from matplotlib.figure import Figure


# pylint: disable=protected-access,too-many-public-methods,too-many-lines,import-outside-toplevel,cyclic-import
class HealpixDataset(Dataset):
    """LSDB Catalog to perform analysis of sky catalogs and efficient spatial operations."""

    hc_structure: HCHealpixDataset

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: HCHealpixDataset,
        loading_config: HatsLoadingConfig | None = None,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.open_...` methods

        Parameters
        ----------
        ddf: nd.NestedFrame
            Dask Nested DataFrame with the source data of the catalog
        ddf_pixel_map: DaskDFPixelMap
            Dictionary mapping HEALPix order and pixel to partition index of ddf
        hc_structure: HCHealpixDataset
            Object with hats metadata of the catalog
        loading_config: HatsLoadingConfig or None, default None
            The configuration used to read the catalog from disk
        """
        super().__init__(ddf, hc_structure, loading_config=loading_config)
        self._ddf_pixel_map = ddf_pixel_map

    def __getitem__(self, item):
        """Select a column or columns from the catalog."""
        # The number of types with which multiple columns can be specified
        # is extensive, so it's safer to check only those type configurations
        # which are known to be either a column name or sequence thereof.
        if isinstance(item, str):
            self._check_unloaded_columns([item])
        elif isinstance(item, Sequence):
            self._check_unloaded_columns([col for col in item if isinstance(col, str)])
        result = self._ddf.__getitem__(item)
        if isinstance(result, nd.NestedFrame):
            return self._create_updated_dataset(ddf=result)
        return result

    def __len__(self):
        """The number of rows in the catalog.

        Returns
        -------
        int
            The number of rows in the catalog, as specified in its metadata.
            This value is undetermined when the catalog is modified, and
            therefore an error is raised.
        """
        return len(self.hc_structure)

    @property
    def nested_columns(self) -> list[str]:
        """The names of the columns of the catalog that are nested.

        Returns
        -------
        list[str]
            The list of nested columns in the catalog.
        """
        return self._ddf.nested_columns

    def _repr_data(self):
        meta = self._ddf._meta
        index = self._repr_divisions
        cols = meta.columns
        if len(cols) == 0:
            series_df = pd.DataFrame([[]] * len(index), columns=cols, index=index)
        else:
            series_df = pd.concat([_repr_data_series(s, index=index) for _, s in meta.items()], axis=1)
        return series_df

    @property
    def _repr_divisions(self):
        pixels = self.get_ordered_healpix_pixels()
        name = f"npartitions={len(pixels)}"
        # Dask will raise an exception, preventing display,
        # if the index does not have at least one element.
        if len(pixels) == 0:
            pixels = ["Empty Catalog"]
        divisions = pd.Index(pixels, name=name)
        return divisions

    def _create_modified_hc_structure(
        self, hc_structure=None, updated_schema=None, **kwargs
    ) -> HCHealpixDataset:
        """Copy the catalog structure and override the specified catalog info parameters."""
        if hc_structure is None:
            hc_structure = self.hc_structure
        return hc_structure.__class__(
            catalog_info=hc_structure.catalog_info.copy_and_update(**kwargs),
            pixels=hc_structure.pixel_tree,
            catalog_path=hc_structure.catalog_path,
            schema=hc_structure.schema if updated_schema is None else updated_schema,
            original_schema=hc_structure.original_schema,
            moc=hc_structure.moc,
        )

    def _create_updated_dataset(
        self,
        ddf: nd.NestedFrame | None = None,
        ddf_pixel_map: DaskDFPixelMap | None = None,
        hc_structure: HCHealpixDataset | None = None,
        updated_catalog_info_params: dict | None = None,
    ) -> Self:
        """Creates a new copy of the catalog, updating any provided arguments

        Shallow copies the ddf and ddf_pixel_map if not provided. Creates a new hc_structure if not provided.
        Updates the hc_structure with any provided catalog info parameters, resets the total rows, removes
        any default columns that don't exist, and updates the pyarrow schema to reflect the new ddf.

        Parameters
        ----------
        ddf : nd.NestedFrame or None, default None
            The catalog ddf to update in the new catalog
        ddf_pixel_map : DaskDFPixelMap or None, default None
            The partition to healpix pixel map to update in the new catalog
        hc_structure : HCHealpixDataset or None, default None
            The hats HealpixDataset object to update in the new catalog
        updated_catalog_info_params : dict or None, default None
            The dictionary of updates to the parameters of the hats dataset object's catalog_info

        Returns
        -------
        Self
            A new dataset object with the arguments updated to those provided to the function, and the
            hc_structure metadata updated to match the new ddf
        """
        ddf = ddf if ddf is not None else self._ddf
        ddf_pixel_map = ddf_pixel_map if ddf_pixel_map is not None else self._ddf_pixel_map
        hc_structure = hc_structure if hc_structure is not None else self.hc_structure
        updated_catalog_info_params = updated_catalog_info_params or {}
        if (
            "default_columns" not in updated_catalog_info_params
            and hc_structure.catalog_info.default_columns is not None
        ):
            updated_catalog_info_params["default_columns"] = [
                col for col in hc_structure.catalog_info.default_columns if col in ddf.columns
            ]
        if "total_rows" not in updated_catalog_info_params:
            updated_catalog_info_params["total_rows"] = None
        updated_schema = get_arrow_schema(ddf)
        hc_structure = self._create_modified_hc_structure(
            hc_structure=hc_structure, updated_schema=updated_schema, **updated_catalog_info_params
        )
        return self.__class__(ddf, ddf_pixel_map, hc_structure, loading_config=self.loading_config)

    def get_healpix_pixels(self) -> list[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog

        Returns
        -------
        list[HealpixPixel]
            List of all Healpix pixels in the catalog
        """
        return self.hc_structure.get_healpix_pixels()

    def get_ordered_healpix_pixels(self) -> list[HealpixPixel]:
        """Get all HEALPix pixels that are contained in the catalog,
        ordered by breadth-first nested ordering.

        Returns
        -------
        list[HealpixPixel]
            List of all Healpix pixels in the catalog
        """
        pixels = self.get_healpix_pixels()
        return np.array(pixels)[get_pixel_argsort(pixels)]

    def aggregate_column_statistics(
        self,
        use_default_columns: bool = True,
        exclude_hats_columns: bool = True,
        exclude_columns: list[str] | None = None,
        include_columns: list[str] | None = None,
        include_pixels: list[HealpixPixel] | None = None,
    ) -> pd.DataFrame:
        """Read footer statistics in parquet metadata, and report on global min/max values.

        Parameters
        ----------
        use_default_columns : bool, default True
            Should we use only the columns that are loaded by default (will be set in the
            metadata by the catalog provider). Defaults to True.
        exclude_hats_columns : bool, default True
            Exclude HATS spatial and partitioning fields from the statistics. Defaults to True.
        exclude_columns : list[str] or None, default None
            Additional columns to exclude from the statistics.
        include_columns : list[str] or None, default None
            If specified, only return statistics for the column
            names provided. Defaults to None, and returns all non-hats columns.
        include_pixels : list[HealpixPixel] or None, default None
            If specified, only return statistics for the pixels indicated. Defaults to none,
            and returns all pixels.

        Returns
        -------
        pd.DataFrame
            Dataframe with global summary statistics
        """
        if use_default_columns and include_columns is None:
            include_columns = self.hc_structure.catalog_info.default_columns

        return self.hc_structure.aggregate_column_statistics(
            exclude_hats_columns=exclude_hats_columns,
            exclude_columns=exclude_columns,
            include_columns=include_columns,
            include_pixels=include_pixels,
        )

    def per_pixel_statistics(
        self,
        use_default_columns: bool = True,
        exclude_hats_columns: bool = True,
        exclude_columns: list[str] | None = None,
        include_columns: list[str] | None = None,
        include_stats: list[str] | None = None,
        multi_index=False,
        include_pixels: list[HealpixPixel] | None = None,
    ) -> pd.DataFrame:
        """Read footer statistics in parquet metadata, and report on
        min/max values for for each data partition.

        Parameters
        ----------
        use_default_columns : bool, default True
            Should we use only the columns that are loaded by default (will be set in the
            metadata by the catalog provider). Defaults to True.
        exclude_hats_columns : bool, default True
            Exclude HATS spatial and partitioning fields from the statistics. Defaults to True.
        exclude_columns : list[str] or None, default None
            Additional columns to exclude from the statistics.
        include_columns : list[str] or None, default None
            If specified, only return statistics for the column
            names provided. Defaults to None, and returns all non-hats columns.
        include_stats : list[str] or None, default None
            If specified, only return the kinds of values from list (min_value, max_value,
            null_count, row_count). Defaults to None, and returns all values.
        multi_index : bool, default False
            Should the returned frame be created with a multi-index, first on
            pixel, then on column name? Default is False, and instead indexes on pixel, with
            separate columns per-data-column and stat value combination.
        include_pixels : list[HealpixPixel] or None, default None
            If specified, only return statistics for the pixels indicated. Defaults to none,
            and returns all pixels.

        Returns
        -------
        pd.Dataframe
            Dataframe with granular per-pixel statistics
        """
        if use_default_columns and include_columns is None:
            include_columns = self.hc_structure.catalog_info.default_columns

        return self.hc_structure.per_pixel_statistics(
            exclude_hats_columns=exclude_hats_columns,
            exclude_columns=exclude_columns,
            include_columns=include_columns,
            include_stats=include_stats,
            multi_index=multi_index,
            include_pixels=include_pixels,
        )

    def get_partition(self, order: int, pixel: int) -> nd.NestedFrame:
        """Get the dask partition for a given HEALPix pixel

        Parameters
        ----------
        order : int
            Order of HEALPix pixel
        pixel : int
            HEALPix pixel number in NESTED ordering scheme

        Returns
        -------
        nd.NestedFrame
            Dask Dataframe with a single partition with data at that pixel

        Raises
        ------
        ValueError
            If no data exists for the specified pixel
        """
        partition_index = self.get_partition_index(order, pixel)
        return self._ddf.partitions[partition_index]

    def get_partition_index(self, order: int, pixel: int) -> int:
        """Get the dask partition for a given HEALPix pixel

        Parameters
        ----------
        order: int
            Order of HEALPix pixel
        pixel: int
            HEALPix pixel number in NESTED ordering scheme

        Returns
        -------
        int
            The index of the partition with data at that pixel

        Raises
        ------
        ValueError
            If no data exists for the specified pixel
        """
        hp_pixel = HealpixPixel(order, pixel)
        if hp_pixel not in self._ddf_pixel_map:
            raise ValueError(f"Pixel at order {order} pixel {pixel} not in Catalog")
        partition_index = self._ddf_pixel_map[hp_pixel]
        return partition_index

    @property
    def partitions(self):
        """Returns the partitions of the catalog"""
        return PartitionIndexer(self)

    @property
    def npartitions(self):
        """Returns the number of partitions of the catalog"""
        return len(self.get_healpix_pixels())

    def head(self, n: int = 5) -> npd.NestedFrame:
        """Returns a few rows of initial data for previewing purposes.

        Parameters
        ----------
        n : int, default 5
            The number of desired rows.

        Returns
        -------
        npd.NestedFrame
            A NestedFrame with up to `n` rows of data.
        """
        dfs = []
        remaining_rows = n
        for partition in self._ddf.partitions:
            if remaining_rows == 0:
                break
            partition_head = partition.head(remaining_rows)
            if len(partition_head) > 0:
                dfs.append(partition_head)
                remaining_rows -= len(partition_head)
        if len(dfs) > 0:
            return npd.NestedFrame(pd.concat(dfs))
        return self._ddf._meta

    def tail(self, n: int = 5) -> npd.NestedFrame:
        """Returns a few rows of data from the end of the catalog for previewing purposes.

        Parameters
        ----------
        n : int, default 5
            The number of desired rows.

        Returns
        -------
        npd.NestedFrame
            A NestedFrame with up to `n` rows of data.
        """
        dfs = []
        remaining_rows = n
        for partition in self._ddf.partitions:
            if remaining_rows == 0:
                break
            partition_tail = partition.tail(remaining_rows)
            if len(partition_tail) > 0:
                dfs.append(partition_tail)
                remaining_rows -= len(partition_tail)
        if len(dfs) > 0:
            return npd.NestedFrame(pd.concat(dfs))
        return self._ddf._meta

    def sample(self, partition_id: int, n: int = 5, seed: int | None = None) -> npd.NestedFrame:
        """Returns a few randomly sampled rows from a given partition.

        Parameters
        ----------
        partition_id : int
            The partition to sample.
        n : int, default 5
            The number of desired rows.
        seed : int or None, default None
            Random seed

        Returns
        -------
        npd.NestedFrame
            A NestedFrame with up to `n` rows of data.

        Notes
        -----
        As with `NestedFrame.sample`, `n` is an approximate number of
        items to return.  The exact number of elements selected will
        depend on how your data is partitioned.  (In practice, it
        should be pretty close.)

        The `seed` argument is passed directly to `random.seed` in order
        to assist with creating predictable outputs when wanted, such
        as in unit tests.
        """
        random.seed(seed)
        # Get the number of partitions so that we can range-check the input argument
        npartitions = len(self.get_healpix_pixels())
        if not 0 <= partition_id < npartitions:
            raise IndexError(f"{partition_id} is out of range [0, {npartitions})")
        partition = self._ddf.partitions[partition_id]
        # Get the count of rows in the partition.
        pixel_rows = len(partition)
        if pixel_rows == 0:
            fraction = 0.0
            logging.debug("Zero rows in partition %d, returning empty", partition_id)
            return partition._meta
        fraction = n / pixel_rows
        logging.debug("Getting %d / %d = %f", n, pixel_rows, fraction)
        return partition.sample(frac=fraction).compute()

    def random_sample(self, n: int = 5, seed: int | None = None) -> npd.NestedFrame:
        """Returns a few randomly sampled rows, like self.sample(), except that
        it randomly samples all partitions in order to fulfill the rows.

        Parameters
        ----------
        n : int, default 5
            The number of desired rows.
        seed : int or None, default None
            Random seed

        Returns
        -------
        npd.NestedFrame
            A NestedFrame with up to `n` rows of data.

        Notes
        -----
        As with `.sample`, `n` is an approximate number of items to
        return.  The exact number of elements selected will depend on
        how your data is partitioned.  (In practice, it should be
        pretty close.)

        The `seed` argument is passed directly to `random.seed` in order
        to assist with creating predictable outputs when wanted, such
        as in unit tests.
        """
        random.seed(seed)
        dfs = []
        if self.hc_structure.catalog_info.total_rows is not None:
            stats = self.hc_structure.per_pixel_statistics()
            # These stats are one *row* per pixel.  The number of
            # columns is permuted, with names like "colname:
            # row_count".  We only need one representative column.
            # Assume the first column is satisfactory.
            rep_col = self.columns[0]
            row_counts = stats[f"{rep_col}: row_count"].map(int)
        else:
            row_counts = np.array(dask.compute(*[dp.shape[0].to_delayed() for dp in self._ddf.partitions]))
        rows_per_partition = np.random.multinomial(n, row_counts / row_counts.sum())
        # With this breakdown, we randomly sample rows from each partition
        # to collect the entire sampling.
        # Logic is borrowed from self.sample(), but we already have a full list
        # of row counts, so we can avoid a lot of overhead.
        for i, (rows, part, part_rows) in enumerate(
            zip(rows_per_partition, self._ddf.partitions, row_counts)
        ):
            if not rows:
                continue
            fraction = rows / part_rows
            logging.debug("Sampling %d / %d rows from partition %d", rows, part_rows, i)
            selection = part.sample(frac=fraction)
            dfs += selection.to_delayed()
        if len(dfs) > 0:
            return npd.NestedFrame(pd.concat(dask.compute(*dfs)))
        return self._ddf._meta

    def query(self, expr: str) -> Self:
        """Filters catalog using a complex query expression.

        Parameters
        ----------
        expr : str
            Query expression to evaluate. The column names that are not valid Python
            variables names should be wrapped in backticks, and any variable values can be
            injected using f-strings. The use of '@' to reference variables is not supported.
            More information about pandas query strings is available
            `here <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`__.

        Returns
        -------
        Self
            A catalog that contains the data from the original catalog that complies
            with the query expression
        """
        ndf = self._ddf.query(expr)
        return self._create_updated_dataset(ddf=ndf)

    def rename(self, columns: Renamer) -> Self:
        """Renames catalog columns (not indices) using a dictionary or function mapping.

        Parameters
        ----------
        columns : dict-like or function
            Transformations to apply to column names.

        Returns
        -------
        Self
            A catalog that contains the data from the original catalog with renamed columns.
        """
        ndf = self._ddf.rename(columns=columns)
        return self._create_updated_dataset(ddf=ndf)

    def cone_search(self, ra: float, dec: float, radius_arcsec: float, fine: bool = True) -> Self:
        """Perform a cone search to filter the catalog.

        Filters to points within radius great circle distance to the point specified by ra and dec in degrees.
        Filters partitions in the catalog to those that have some overlap with the cone.

        Parameters
        ----------
        ra : float
            Right Ascension of the center of the cone in degrees
        dec : float
            Declination of the center of the cone in degrees
        radius_arcsec : float
            Radius of the cone in arcseconds
        fine : bool, default True
            True if points are to be filtered, False if not. Defaults to True.

        Returns
        -------
        Self
            A new Catalog containing the points filtered to those within the cone, and the partitions that
            overlap the cone.
        """
        return self.search(ConeSearch(ra, dec, radius_arcsec, fine))

    def box_search(self, ra: tuple[float, float], dec: tuple[float, float], fine: bool = True) -> Self:
        """Performs filtering according to right ascension and declination ranges.

        The right ascension edges follow great arc circles and the declination edges
        follow small arc circles.

        Filters to points within the region specified in degrees.
        Filters partitions in the catalog to those that have some overlap with the region.

        Parameters
        ----------
        ra : tuple[float, float]
            The right ascension minimum and maximum values.
        dec : tuple[float, float]
            The declination minimum and maximum values.
        fine : bool, default True
            True if points are to be filtered, False if not. Defaults to True.

        Returns
        -------
        Self
            A new catalog containing the points filtered to those within the region, and the
            partitions that have some overlap with it.
        """
        return self.search(BoxSearch(ra, dec, fine))

    def polygon_search(self, vertices: list[tuple[float, float]], fine: bool = True) -> Self:
        """Perform a polygonal search to filter the catalog.

        IMPORTANT: Requires additional ``lsst-sphgeom`` package

        Filters to points within the polygonal region specified in ra and dec, in degrees.
        Filters partitions in the catalog to those that have some overlap with the region.

        Parameters
        ----------
        vertices : list[tuple[float, float]]
            The list of vertices of the polygon to filter pixels with, as a list
            of (ra,dec) coordinates, in degrees.
        fine : bool, default True
            True if points are to be filtered, False if not. Defaults to True.

        Returns
        -------
        Self
            A new catalog containing the points filtered to those within the
            polygonal region, and the partitions that have some overlap with it.
        """
        return self.search(PolygonSearch(vertices, fine))

    def order_search(self, min_order: int = 0, max_order: int | None = None) -> Self:
        """Filter catalog by order of HEALPix.

        Parameters
        ----------
        min_order : int, default 0
            Minimum HEALPix order to select. Defaults to 0.
        max_order : int or None, default None
            Maximum HEALPix order to select. Defaults to maximum catalog order.

        Returns
        -------
        Self
            A new Catalog containing only the pixels of orders specified (inclusive).
        """
        return self.search(OrderSearch(min_order, max_order))

    def pixel_search(
        self, pixels: tuple[int, int] | HealpixPixel | list[tuple[int, int] | HealpixPixel]
    ) -> Self:
        """Finds all catalog pixels that overlap with the requested pixel set.

        Parameters
        ----------
        pixels : list[tuple[int, int]]
            The list of HEALPix tuples (order, pixel) that define the region for the search.

        Returns
        -------
        Self
            A new Catalog containing only the pixels that overlap with the requested pixel set.
        """
        return self.search(PixelSearch(pixels))

    def moc_search(self, moc: MOC, fine: bool = True) -> Self:
        """Finds all catalog points that are contained within a moc.

        Parameters
        ----------
        moc : mocpy.MOC
            The moc that defines the region for the search.
        fine : bool, default True
            True if points are to be filtered, False if only partitions. Defaults to True.

        Returns
        -------
        Self
            A new Catalog containing only the points that are within the moc.
        """
        return self.search(MOCSearch(moc, fine=fine))

    def _perform_search(
        self,
        metadata: HCHealpixDataset,
        search: AbstractSearch,
    ) -> tuple[DaskDFPixelMap, nd.NestedFrame]:
        """Performs a search on the catalog from a list of pixels to search in

        Parameters
        ----------
        metadata : HCHealpixDataset
            The metadata of the hats catalog after the coarse filtering is applied.
            The partitions it contains are only those that overlap with the spatial region.
        search : AbstractSearch
            Instance of AbstractSearch.

        Returns
        -------
        tuple[DaskDFPixelMap, nd.NestedFrame]
            A tuple containing a dictionary mapping pixel to partition index and a dask dataframe
            containing the search results
        """
        filtered_pixels = metadata.get_healpix_pixels()
        if len(filtered_pixels) == 0:
            return {}, nd.NestedFrame.from_single_partition(self._ddf._meta)
        target_partitions_indices = [self._ddf_pixel_map[pixel] for pixel in filtered_pixels]
        filtered_partitions_ddf = self._ddf.partitions[target_partitions_indices]
        if search.fine:
            filtered_partitions_ddf = filtered_partitions_ddf.map_partitions(
                search.search_points,
                metadata.catalog_info,
                meta=self._ddf._meta,
                transform_divisions=False,
            )
        ddf_partition_map = {pixel: i for i, pixel in enumerate(filtered_pixels)}
        return ddf_partition_map, filtered_partitions_ddf

    def search(self, search: AbstractSearch) -> Self:
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria.
        Filters to points that match some finer criteria.

        Parameters
        ----------
        search : AbstractSearch
            Instance of AbstractSearch.

        Returns
        -------
        Self
            A new Catalog containing the points filtered to those matching the search parameters.
        """
        if (
            self.hc_structure.catalog_info.total_rows is not None
            and self.hc_structure.catalog_base_dir is not None
            and self.hc_structure.original_schema is not None
        ):
            return self._reload_with_filter(search)
        filtered_hc_structure = search.filter_hc_catalog(self.hc_structure)
        ddf_partition_map, search_ndf = self._perform_search(filtered_hc_structure, search)
        return self._create_updated_dataset(
            ddf=search_ndf, ddf_pixel_map=ddf_partition_map, hc_structure=filtered_hc_structure
        )

    def _reload_with_filter(self, search: AbstractSearch):
        """Reloads the catalog from storage, applying a given search filter.

        Uses the columns of the current catalog to select the same columns to reload."""
        from lsdb.loaders.hats.read_hats import _load_catalog

        all_columns = self._ddf._meta.all_columns
        base_columns = all_columns.pop("base", [])
        nonnested_basecols = [c for c in base_columns if c not in self._ddf._meta.nested_columns]
        loading_columns = [
            [f"{base_name}.{col}" for col in all_columns[base_name]] for base_name in all_columns
        ]
        columns = nonnested_basecols + [c for cs in loading_columns for c in cs]

        hc_structure = self._create_modified_hc_structure(updated_schema=self.hc_structure.original_schema)
        new_loading_config = HatsLoadingConfig(
            search_filter=search,
            columns=columns,
            margin_cache=None,
            error_empty_filter=False,
        )
        if self.loading_config:
            new_loading_config.filters = self.loading_config.filters
            new_loading_config.kwargs = self.loading_config.kwargs
            new_loading_config.path_generator = self.loading_config.path_generator

        return _load_catalog(hc_structure, new_loading_config)

    def map_partitions(
        self,
        func: Callable[..., npd.NestedFrame],
        *args,
        meta: pd.DataFrame | pd.Series | dict | Iterable | tuple | None = None,
        include_pixel: bool = False,
        **kwargs,
    ) -> Self | dd.Series:
        """Applies a function to each partition in the catalog.

        The ra and dec of each row is assumed to remain unchanged.

        Parameters
        ----------
        func : Callable
            The function applied to each partition, which will be called with:
            `func(partition: npd.NestedFrame, *args, **kwargs)` with the additional args and kwargs passed
            to the `map_partitions` function. If the `include_pixel` parameter is set, the function will
            be called with the `healpix_pixel` as the second positional argument set to the healpix pixel
            of the partition as
            `func(partition: npd.NestedFrame, healpix_pixel: HealpixPixel, *args, **kwargs)`
        *args
            Additional positional arguments to call `func` with.
        meta : pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None, default None
            An empty pandas DataFrame that has columns matching the output of the function applied to a
            partition. Other types are accepted to describe the output dataframe format, for full details
            see the dask documentation https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
            If meta is None (default), LSDB will try to work out the output schema of the function by
            calling the function with an empty DataFrame. If the function does not work with an empty
            DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
            will generate empty partitions, though these can be removed by calling the
            `Catalog.prune_empty_partitions` method.
        include_pixel : bool, default False
            Whether to pass the Healpix Pixel of the partition as a `HealpixPixel`
            object to the second positional argument of the function
        **kwargs
            Additional keyword args to pass to the function. These are passed to the Dask DataFrame
            `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
            `transform_divisions` will be passed through and work as described in the dask documentation
            https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html

        Returns
        -------
        Self or dd.Series
            A new catalog with each partition replaced with the output of the function applied to the original
            partition. If the function returns a non dataframe output, a dask Series will be returned.
        """
        if meta is None:
            if include_pixel:
                meta = func(self._ddf._meta.copy(), HealpixPixel(0, 0), *args, **kwargs)
            else:
                meta = func(self._ddf._meta.copy(), *args, **kwargs)
            if meta is None:
                raise ValueError(
                    "func returned None for empty DataFrame input. The function must return a value, changing"
                    " the partitions in place will not work. If the function does not work for empty inputs, "
                    "please specify a `meta` argument."
                )
        if include_pixel:
            pixels = self.get_ordered_healpix_pixels()

            def apply_func(df, *args, partition_info=None, **kwargs):
                """Uses `partition_info` passed by dask `map_partitions` to get healpix pixel to pass to
                ufunc"""
                assert partition_info is not None, "partition_info must be provided by dask map_partitions"
                partition_number = partition_info["number"]
                pixel = pixels[partition_number]
                try:
                    return func(df, pixel, *args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Error applying function {func.__name__} to partition "
                        f"{partition_number}, pixel {pixel}: {e}"
                    ) from e

            output_ddf = self._ddf.map_partitions(apply_func, *args, meta=meta, **kwargs)
        else:

            def apply_func(df, *args, partition_info=None, **kwargs):
                """Applies the function to the partition without pixel information"""
                assert partition_info is not None, "partition_info must be provided by dask map_partitions"
                partition_number = partition_info["number"]
                try:
                    return func(df, *args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Error applying function {func.__name__} to partition {partition_number}: {str(e)}"
                    ) from e

            output_ddf = self._ddf.map_partitions(apply_func, *args, meta=meta, **kwargs)

        if isinstance(output_ddf, nd.NestedFrame) | isinstance(output_ddf, dd.DataFrame):
            return self._create_updated_dataset(ddf=nd.NestedFrame.from_dask_dataframe(output_ddf))
        warnings.warn(
            "output of the function must be a DataFrame to generate an LSDB `Catalog`. `map_partitions` "
            "will return a dask object instead of a Catalog.",
            RuntimeWarning,
        )
        return output_ddf

    def prune_empty_partitions(self, persist: bool = False) -> Self:
        """Prunes the catalog of its empty partitions

        Parameters
        ----------
        persist : bool, default False
            If True previous computations are saved. Defaults to False.

        Returns
        -------
        Self
            A new catalog containing only its non-empty partitions
        """
        warnings.warn("Pruning empty partitions is expensive. It may run slow!", RuntimeWarning)
        if persist:
            self._ddf.persist()
        non_empty_pixels, non_empty_partitions = self._get_non_empty_partitions()
        search_ddf = (
            self._ddf.partitions[non_empty_partitions]
            if len(non_empty_partitions) > 0
            else nd.NestedFrame.from_single_partition(self._ddf._meta)
        )
        ddf_partition_map = {pixel: i for i, pixel in enumerate(non_empty_pixels)}
        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(non_empty_pixels)
        return self._create_updated_dataset(
            ddf=search_ddf, ddf_pixel_map=ddf_partition_map, hc_structure=filtered_hc_structure
        )

    def _get_non_empty_partitions(self) -> tuple[list[HealpixPixel], np.ndarray]:
        """Determines which pixels and partitions of a catalog are not empty

        Returns
        -------
        tuple[list[HealpixPixel], np.ndarray]
            A tuple with the non-empty pixels and respective partitions
        """

        # Compute partition lengths (expensive operation)
        partition_sizes = self._ddf.map_partitions(len).compute()
        non_empty_partition_indices = np.argwhere(partition_sizes > 0).flatten()

        non_empty_indices_set = set(non_empty_partition_indices)

        # Extract the non-empty pixels and respective partitions
        non_empty_pixels = []
        for pixel, partition_index in self._ddf_pixel_map.items():
            if partition_index in non_empty_indices_set:
                non_empty_pixels.append(pixel)

        return non_empty_pixels, non_empty_partition_indices

    def plot_pixels(self, projection: str = "MOL", **kwargs) -> tuple[Figure, WCSAxes]:
        """Create a visual map of the pixel density of the catalog.

        Parameters
        ----------
        projection : str, default "MOL"
            The map projection to use. Available projections listed at
            https://docs.astropy.org/en/stable/wcs/supported_projections.html
        **kwargs
            Additional keyword arguments to pass to plotting call.

        Returns
        -------
        tuple[Figure, WCSAxes]
        """
        return self.hc_structure.plot_pixels(projection=projection, **kwargs)

    def plot_coverage(self, **kwargs) -> tuple[Figure, WCSAxes]:
        """Create a visual map of the coverage of the catalog.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to hats.Catalog.plot_moc

        Returns
        -------
        tuple[Figure, WCSAxes]
        """
        return self.hc_structure.plot_moc(**kwargs)

    @deprecated(
        version="0.7.3", reason="`to_hats` will be removed in the future, " "use `write_catalog` instead."
    )
    def to_hats(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        overwrite: bool = False,
        error_if_empty: bool = True,
        **kwargs,
    ):
        """Save the catalog to disk in the HATS format. See write_catalog()."""
        self.write_catalog(
            base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            overwrite=overwrite,
            error_if_empty=error_if_empty,
            **kwargs,
        )

    def write_catalog(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        overwrite: bool = False,
        error_if_empty: bool = True,
        **kwargs,
    ):
        """Save the catalog to disk in HATS format.

        Parameters
        ----------
        base_catalog_path : str | Path | UPath
            Location where catalog is saved to
        catalog_name : str or None, default None
            The name of the catalog to be saved
        default_columns : list[str] or None, default None
            A metadata property with the list of the columns in the catalog to
            be loaded by default. By default, uses the default columns from the
            original hats catalogs if they exist.
        overwrite : bool, default False
            If True existing catalog is overwritten
        error_if_empty : bool, default True
            If True, raises an error if the catalog is empty.
        **kwargs
            Arguments to pass to the parquet write operations
        """
        self._check_unloaded_columns(default_columns)
        io.to_hats(
            self,
            base_catalog_path=base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            overwrite=overwrite,
            error_if_empty=error_if_empty,
            **kwargs,
        )

    def nest_lists(
        self,
        base_columns: list[str] | None = None,
        list_columns: list[str] | None = None,
        name: str = "nested",
    ) -> Self:  # type: ignore[name-defined] # noqa: F821:
        """Creates a new catalog with a set of list columns packed into a nested column.

        Parameters
        ----------
        base_columns : list-like or None, default None
            Any columns that have non-list values in the input catalog.
            These will simply be kept as identical columns in the result. If None, is inferred to be
            all columns in the input catalog that are not considered list-value columns.
        list_columns : list-like or None, default None
            The list-value columns that should be packed into a nested
            column. All columns in the list will attempt to be packed into a single nested column
            with the name provided in `nested_name`. All columns in list_columns must have pyarrow
            list dtypes, otherwise the operation will fail. If None, is defined as all columns not in
            `base_columns`.
        name : str, default "nested"
            The name of the output column the `nested_columns` are packed into.

        Returns
        -------
        Self
            A new catalog with specified list columns nested into a new nested column.

        Notes
        -----
            As noted above, all columns in `list_columns` must have a pyarrow
            ListType dtype. This is needed for proper meta propagation. To convert
            a list column to this dtype, you can use this command structure:
            `nf= nf.astype({"colname": pd.ArrowDtype(pa.list_(pa.int64()))})`
            Where pa.int64 above should be replaced with the correct dtype of the
            underlying data accordingly. Additionally, it's a known issue in Dask
            (https://github.com/dask/dask/issues/10139) that columns with list
            values will by default be converted to the string type. This will
            interfere with the ability to recast these to pyarrow lists. We
            recommend setting the following dask config setting to prevent this:
            `dask.config.set({"dataframe.convert-string":False})`
        """
        self._check_unloaded_columns(base_columns)
        self._check_unloaded_columns(list_columns)
        new_ddf = nd.NestedFrame.from_lists(
            self._ddf,
            base_columns=base_columns,
            list_columns=list_columns,
            name=name,
        )
        return self._create_updated_dataset(ddf=new_ddf)

    def map_rows(
        self,
        func,
        columns=None,
        row_container="dict",
        output_names=None,
        infer_nesting=True,
        append_columns=False,
        meta=None,
        **kwargs,
    ) -> Self:
        """Takes a function and applies it to each top-level row of the Catalog.

        docstring copied from nested-pandas

        Nested columns are packaged alongside base columns and available for function use, where base columns
        are passed as scalars and nested columns are passed as numpy arrays. The way in which the row data is
        packaged is configurable (by default, a dictionary) and controlled by the `row_container` argument.

        Parameters
        ----------
        func : callable
            Function to apply to each nested dataframe. The first arguments to `func` should be which
            columns to apply the function to. See the Notes for recommendations
            on writing func outputs.
        columns : None | str | list of str, default None
            Specifies which columns to pass to the function in the row_container format.
            If None, all columns are passed. If list of str, those columns are passed.
            If str, a single column is passed or if the string is a nested column, then all nested sub-columns
            are passed (e.g. columns="nested" passes all columns of the nested dataframe "nested"). To pass
            individual nested sub-columns, use the hierarchical column name (e.g. columns=["nested.t",...]).
        row_container : 'dict' or 'args', default 'dict'
            Specifies how the row data will be packaged when passed as an input to the function.
            If 'dict', the function will be called as `func({"col1": value, ...}, **kwargs)`, so func should
            expect a single dictionary input with keys corresponding to column names.
            If 'args', the function will be called as `func(value, ..., **kwargs)`, so func should expect
            positional arguments corresponding to the columns specified in `args`.
        output_names : None | str | list of str, default None
            Specifies the names of the output columns in the resulting NestedFrame. If None, the function
            will return whatever names the user function returns. If specified will override any names
            returned by the user function provided the number of names matches the number of outputs. When not
            specified and the user function returns values without names (e.g. a list or tuple), the output
            columns will be enumerated (e.g. "0", "1", ...).
        infer_nesting : bool, default True
            If True, the function will pack output columns into nested
            structures based on column names adhering to a nested naming
            scheme. E.g. "nested.b" and "nested.c" will be packed into a column
            called "nested" with columns "b" and "c". If False, all outputs
            will be returned as base columns. Note that this will trigger off of names specified in
            `output_names` in addition to names returned by the user function.
        append_columns : bool, default False
            if True, the output columns should be appended to those in the original NestedFrame.
        meta : dataframe or series-like, default None
            The dask meta of the output. If append_columns is True, the meta should specify just the
            additional columns output by func.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `HealpixDataset`
            `HealpixDataset` with the results of the function applied to the columns of the frame.

        Notes
        -----
        If concerned about performance, specify `columns` to only include the columns
        needed for the function, as this will avoid the overhead of packaging
        all columns for each row.

        By default, `map_rows` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. It's recommended
        to either specify `output_names` or have `func` return a dictionary
        where each key is an output column of the dataframe returned by
        `map_rows` (as shown above).
        Examples
        --------

        Writing a function that takes a row as a dictionary:

        >>> import numpy as np
        >>> import lsdb
        >>> import pandas as pd
        >>> catalog = lsdb.from_dataframe(pd.DataFrame({"ra":[0, 10], "dec":[5, 15],
        ...                                             "mag":[21, 22], "mag_err":[.1, .2]}))

        >>> def my_sigma(row):
        ...    '''map_rows will return a NestedFrame with two columns'''
        ...    return row["mag"] + row["mag_err"], row["mag"] - row["mag_err"]
        >>> meta = {"plus_one": np.float64, "minus_one": np.float64}
        >>> catalog.map_rows(my_sigma,
        ...                  columns=["mag","mag_err"],
        ...                  output_names=["plus_one", "minus_one"],
        ...                  meta=meta).compute().reset_index()
                   _healpix_29  plus_one  minus_one
        0  1372475556631677955      21.1       20.9
        1  1389879706834706546      22.2       21.8


        Writing the same function using positional arguments:

        >>> def my_sigma(col1, col2):
        ...    '''map_rows will return a NestedFrame with two columns'''
        ...    return col1 + col2, col1 - col2
        >>> meta = {"plus_one": np.float64, "minus_one": np.float64}
        >>> catalog.map_rows(my_sigma,
        ...                  columns=["mag","mag_err"],
        ...                  row_container='args', # send columns as positional args
        ...                  output_names=["plus_one", "minus_one"],
        ...                  meta=meta).compute().reset_index()
                   _healpix_29  plus_one  minus_one
        0  1372475556631677955      21.1       20.9
        1  1389879706834706546      22.2       21.8

        See more examples in the nested-pandas documentation.
        """
        self._check_unloaded_columns(columns)

        if append_columns:
            meta = concat_metas([self._ddf._meta.copy(), meta])

        catalog_info = self.hc_structure.catalog_info

        def reduce_part(df):
            reduced_result = npd.NestedFrame(df).map_rows(
                func,
                columns=columns,
                row_container=row_container,
                output_names=output_names,
                infer_nesting=infer_nesting,
                **kwargs,
            )
            if append_columns:
                if catalog_info.ra_column in reduced_result or catalog_info.dec_column in reduced_result:
                    raise ValueError("ra and dec columns can not be modified using reduce")
                return npd.NestedFrame(pd.concat([df, reduced_result], axis=1))
            return reduced_result

        ndf = nd.NestedFrame.from_dask_dataframe(self._ddf.map_partitions(reduce_part, meta=meta))

        hc_updates = {}
        if not append_columns:
            hc_updates = {"ra_column": "", "dec_column": ""}
        return self._create_updated_dataset(ddf=ndf, updated_catalog_info_params=hc_updates)

    # pylint: disable=duplicate-code
    def plot_points(
        self,
        *,
        ra_column: str | None = None,
        dec_column: str | None = None,
        color_col: str | None = None,
        projection: str = "MOL",
        title: str | None = None,
        fov: Quantity | tuple[Quantity, Quantity] | None = None,
        center: SkyCoord | None = None,
        wcs: astropy.wcs.WCS | None = None,
        frame_class: Type[BaseFrame] | None = None,
        ax: WCSAxes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        """Plots the points in the catalog as a scatter plot

        Performs a scatter plot on a WCSAxes after computing the points of the catalog.
        This will perform compute on the catalog, and so may be slow/resource intensive.
        If the fov or wcs args are set, only the partitions in the catalog visible to the
        plot will be computed. The scatter points can be colored by a column of the catalog
        by using the `color_col` kwarg.

        Parameters
        ----------
        ra_column : str | None, default None
            The column to use as the RA of the points to plot. Defaults to the
            catalog's default RA column. Useful for plotting joined or cross-matched points
        dec_column : str | None, default None
            The column to use as the Declination of the points to plot. Defaults to
            the catalog's default Declination column. Useful for plotting joined or cross-matched points
        color_col : str | None, default None
            The column to use as the color array for the scatter plot. Allows coloring
            of the points by the values of a given column.
        projection : str, default "MOL"
            The projection to use in the WCS. Available projections listed at
            https://docs.astropy.org/en/stable/wcs/supported_projections.html
        title : str, default None
            The title of the plot
        fov : Quantity or Sequence[Quantity, Quantity] | None, default None
            The Field of View of the WCS. Must be an astropy Quantity with an angular unit, or a tuple of
            quantities for different longitude and latitude FOVs (Default covers the full sky)
        center : SkyCoord | None, default None
            The center of the projection in the WCS (Default: SkyCoord(0, 0))
        wcs : WCS | None, default None
            The WCS to specify the projection of the plot. If used, all other WCS parameters
            are ignored and the parameters from the WCS object is used.
        frame_class : Type[BaseFrame] | None, default None
            The class of the frame for the WCSAxes to be initialized
            with. if the `ax` kwarg is used, this value is ignored (By Default uses EllipticalFrame for
            full sky projection. If FOV is set, RectangularFrame is used)
        ax : WCSAxes | None, default None
            The matplotlib axes to plot onto. If None, an axes will be created to be
            used. If specified, the axes must be an astropy WCSAxes, and the `wcs` parameter must be set
            with the WCS object used in the axes. (Default: None)
        fig : Figure | None, default None
            The matplotlib figure to add the axes to. If None, one will be created,
            unless ax is specified (Default: None)
        **kwargs
            Additional kwargs to pass to creating the matplotlib `scatter` function. These include
            `c` for color, `s` for the size of hte points, `marker` for the maker type, `cmap` and `norm`
            if `color_col` is used

        Returns
        -------
        tuple[Figure, WCSAxes]
            The figure and axes used for the plot
        """
        try:
            # pylint: disable=import-outside-toplevel
            from hats.inspection._plotting import _get_fov_moc_from_wcs, _initialize_wcs_axes
            from matplotlib import pyplot as plt  # pylint: disable=unused-import
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to use this method. Install with pip or conda."
            ) from exc

        fig, ax, wcs = _initialize_wcs_axes(
            projection=projection,
            fov=fov,
            center=center,
            wcs=wcs,
            frame_class=frame_class,
            ax=ax,
            fig=fig,
            figsize=(9, 5),
        )

        fov_moc = _get_fov_moc_from_wcs(wcs)

        computed_catalog = (
            self.search(MOCSearch(fov_moc)).compute() if fov_moc is not None else self.compute()
        )

        if ra_column is None:
            ra_column = self.hc_structure.catalog_info.ra_column
        if dec_column is None:
            dec_column = self.hc_structure.catalog_info.dec_column

        if ra_column is None:
            raise ValueError("Catalog has no RA Column")

        if dec_column is None:
            raise ValueError("Catalog has no DEC Column")

        if title is None:
            title = f"Points in the {self.name} catalog"

        self._check_unloaded_columns([ra_column, dec_column, color_col])
        return plot_points(
            computed_catalog,
            ra_column,
            dec_column,
            color_col=color_col,
            projection=projection,
            title=title,
            fov=fov,
            center=center,
            wcs=wcs,
            frame_class=frame_class,
            ax=ax,
            fig=fig,
            **kwargs,
        )
