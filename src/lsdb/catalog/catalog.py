from __future__ import annotations

from typing import List, Tuple, Type

import hats as hc
import nested_dask as nd
import nested_pandas as npd
import pandas as pd
from hats.catalog.index.index_catalog import IndexCatalog as HCIndexCatalog
from hats.pixel_math.polygon_filter import SphericalCoordinates
from pandas._libs import lib
from pandas._typing import AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default

from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.search import BoxSearch, ConeSearch, IndexSearch, OrderSearch, PolygonSearch
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.core.search.pixel_search import PixelSearch
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data
from lsdb.dask.join_catalog_data import (
    join_catalog_data_nested,
    join_catalog_data_on,
    join_catalog_data_through,
    merge_asof_catalog_data,
)
from lsdb.dask.partition_indexer import PartitionIndexer
from lsdb.io.schema import get_arrow_schema
from lsdb.types import DaskDFPixelMap


# pylint: disable=R0903, W0212
class Catalog(HealpixDataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hats.Catalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.Catalog

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.Catalog,
        margin: MarginCatalog | None = None,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hats.Catalog` object with hats metadata of the catalog
        """
        super().__init__(ddf, ddf_pixel_map, hc_structure)
        self.margin = margin

    @property
    def partitions(self):
        """Returns the partitions of the catalog"""
        return PartitionIndexer(self)

    def head(self, n: int = 5) -> npd.NestedFrame:
        """Returns a few rows of data for previewing purposes.

        Args:
            n (int): The number of desired rows.

        Returns:
            A pandas DataFrame with up to `n` of data.
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

    def query(self, expr: str) -> Catalog:
        catalog = super().query(expr)
        if self.margin is not None:
            catalog.margin = self.margin.query(expr)
        return catalog

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
        algorithm: (
            Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
        ) = BuiltInCrossmatchAlgorithm.KD_TREE,
        output_catalog_name: str | None = None,
        require_right_margin: bool = False,
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
                    - `kd_tree`: find the k-nearest neighbors using a kd_tree

                Custom function:
                    To specify a custom function, write a class that subclasses the
                    `AbstractCrossmatchAlgorithm` class, and overwrite the `perform_crossmatch` function.

                    The function should be able to perform a crossmatch on two pandas DataFrames
                    from a partition from each catalog. It should return two 1d numpy arrays of equal lengths
                    with the indices of the matching rows from the left and right dataframes, and a dataframe
                    with any extra columns generated by the crossmatch algorithm, also with the same length.
                    These columns are specified in {AbstractCrossmatchAlgorithm.extra_columns}, with
                    their respective data types, by means of an empty pandas dataframe. As an example,
                    the KdTreeCrossmatch algorithm outputs a "_dist_arcsec" column with the distance between
                    data points. Its extra_columns attribute is specified as follows::

                        pd.DataFrame({"_dist_arcsec": pd.Series(dtype=np.dtype("float64"))})

                    The class will have been initialized with the following parameters, which the
                    crossmatch function should use:
                        - left: npd.NestedFrame,
                        - right: npd.NestedFrame,
                        - left_order: int,
                        - left_pixel: int,
                        - right_order: int,
                        - right_pixel: int,
                        - left_metadata: hc.catalog.Catalog,
                        - right_metadata: hc.catalog.Catalog,
                        - right_margin_hc_structure: hc.margin.MarginCatalog,
                        - suffixes: Tuple[str, str]

                    You may add any additional keyword argument parameters to the crossmatch
                    function definition, and the user will be able to pass them in as kwargs in the
                    `Catalog.crossmatch` method. Any additional keyword arguments must also be added to the
                    `CrossmatchAlgorithm.validate` classmethod by overwriting the method.

            output_catalog_name (str): The name of the resulting catalog.
                Default: {left_name}_x_{right_name}
            require_right_margin (bool): If true, raises an error if the right margin is missing which could
                lead to incomplete crossmatches. Default: False

        Returns:
            A Catalog with the data from the left and right catalogs merged with one row for each
            pair of neighbors found from cross-matching.

            The resulting table contains all columns from the left and right catalogs with their
            respective suffixes and, whenever specified, a set of extra columns generated by the
            crossmatch algorithm.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")
        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")
        if other.margin is None and require_right_margin:
            raise ValueError("Right catalog margin cache is required for cross-match.")
        if output_catalog_name is None:
            output_catalog_name = f"{self.name}_x_{other.name}"
        ddf, ddf_map, alignment = crossmatch_catalog_data(
            self, other, suffixes, algorithm=algorithm, **kwargs
        )
        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
            catalog_name=output_catalog_name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )
        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf))
        return Catalog(ddf, ddf_map, hc_catalog)

    def cone_search(self, ra: float, dec: float, radius_arcsec: float, fine: bool = True) -> Catalog:
        """Perform a cone search to filter the catalog

        Filters to points within radius great circle distance to the point specified by ra and dec in degrees.
        Filters partitions in the catalog to those that have some overlap with the cone.

        Args:
            ra (float): Right Ascension of the center of the cone in degrees
            dec (float): Declination of the center of the cone in degrees
            radius_arcsec (float): Radius of the cone in arcseconds
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A new Catalog containing the points filtered to those within the cone, and the partitions that
            overlap the cone.
        """
        return self.search(ConeSearch(ra, dec, radius_arcsec, fine))

    def box_search(
        self,
        ra: Tuple[float, float] | None = None,
        dec: Tuple[float, float] | None = None,
        fine: bool = True,
    ) -> Catalog:
        """Performs filtering according to right ascension and declination ranges.

        Filters to points within the region specified in degrees.
        Filters partitions in the catalog to those that have some overlap with the region.

        Args:
            ra (Tuple[float, float]): The right ascension minimum and maximum values.
            dec (Tuple[float, float]): The declination minimum and maximum values.
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A new catalog containing the points filtered to those within the region, and the
            partitions that have some overlap with it.
        """
        return self.search(BoxSearch(ra, dec, fine))

    def polygon_search(self, vertices: List[SphericalCoordinates], fine: bool = True) -> Catalog:
        """Perform a polygonal search to filter the catalog.

        Filters to points within the polygonal region specified in ra and dec, in degrees.
        Filters partitions in the catalog to those that have some overlap with the region.

        Args:
            vertices (List[Tuple[float, float]): The list of vertices of the polygon to
                filter pixels with, as a list of (ra,dec) coordinates, in degrees.
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A new catalog containing the points filtered to those within the
            polygonal region, and the partitions that have some overlap with it.
        """
        return self.search(PolygonSearch(vertices, fine))

    def index_search(self, ids, catalog_index: HCIndexCatalog, fine: bool = True) -> Catalog:
        """Find rows by ids (or other value indexed by a catalog index).

        Filters partitions in the catalog to those that could contain the ids requested.
        Filters to points that have matching values in the id field.

        NB: This requires a previously-computed catalog index table.

        Args:
            ids: Values to search for.
            catalog_index (HCIndexCatalog): A pre-computed hats index catalog.
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A new Catalog containing the points filtered to those matching the ids.
        """
        return self.search(IndexSearch(ids, catalog_index, fine))

    def order_search(self, min_order: int = 0, max_order: int | None = None) -> Catalog:
        """Filter catalog by order of HEALPix.

        Args:
            min_order (int): Minimum HEALPix order to select. Defaults to 0.
            max_order (int): Maximum HEALPix order to select. Defaults to maximum catalog order.

        Returns:
            A new Catalog containing only the pixels of orders specified (inclusive).
        """
        return self.search(OrderSearch(min_order, max_order))

    def pixel_search(self, pixels: List[Tuple[int, int]]) -> Catalog:
        """Finds all catalog pixels that overlap with the requested pixel set.

        Args:
            pixels (List[Tuple[int, int]]): The list of HEALPix tuples (order, pixel)
                that define the region for the search.

        Returns:
            A new Catalog containing only the pixels that overlap with the requested pixel set.
        """
        return self.search(PixelSearch(pixels))

    def search(self, search: AbstractSearch):
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria.
        Filters to points that match some finer criteria.

        Args:
            search (AbstractSearch): Instance of AbstractSearch.

        Returns:
            A new Catalog containing the points filtered to those matching the search parameters.
        """
        filtered_hc_structure = search.filter_hc_catalog(self.hc_structure)
        ddf_partition_map, search_ndf = self._perform_search(filtered_hc_structure, search)
        margin = self.margin.search(search) if self.margin is not None else None
        return Catalog(search_ndf, ddf_partition_map, filtered_hc_structure, margin=margin)

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
    ) -> nd.NestedFrame:
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

    def merge_asof(
        self,
        other: Catalog,
        direction: str = "backward",
        suffixes: Tuple[str, str] | None = None,
        output_catalog_name: str | None = None,
    ):
        """Uses the pandas `merge_asof` function to merge two catalogs on their indices by distance of keys

        Must be along catalog indices, and does not include margin caches, meaning results may be incomplete
        for merging points.

        This function is intended for use in special cases such as Dust Map Catalogs, for general merges,
        the `crossmatch`and `join` functions should be used.

        Args:
            other (lsdb.Catalog): the right catalog to merge to
            suffixes (Tuple[str,str]): the suffixes to apply to each partition's column names
            direction (str): the direction to perform the merge_asof

        Returns:
            A new catalog with the columns from each of the input catalogs with their respective suffixes
            added, and the rows merged using merge_asof on the specified columns.
        """
        if suffixes is None:
            suffixes = (f"_{self.name}", f"_{other.name}")

        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")

        ddf, ddf_map, alignment = merge_asof_catalog_data(self, other, suffixes=suffixes, direction=direction)

        if output_catalog_name is None:
            output_catalog_name = (
                f"{self.hc_structure.catalog_info.catalog_name}_merge_asof_"
                f"{other.hc_structure.catalog_info.catalog_name}"
            )

        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
            catalog_name=output_catalog_name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )

        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf))
        return Catalog(ddf, ddf_map, hc_catalog)

    def join(
        self,
        other: Catalog,
        left_on: str | None = None,
        right_on: str | None = None,
        through: AssociationCatalog | None = None,
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
            through (AssociationCatalog): an association catalog that provides the alignment
                between pixels and individual rows.
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

        if through is not None:
            ddf, ddf_map, alignment = join_catalog_data_through(self, other, through, suffixes)

            if output_catalog_name is None:
                output_catalog_name = self.hc_structure.catalog_info.catalog_name

            new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
                catalog_name=output_catalog_name,
                ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
                dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
            )

            hc_catalog = hc.catalog.Catalog(
                new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf)
            )
            return Catalog(ddf, ddf_map, hc_catalog)
        if left_on is None or right_on is None:
            raise ValueError("Either both of left_on and right_on, or through must be set")
        if left_on not in self._ddf.columns:
            raise ValueError("left_on must be a column in the left catalog")

        if right_on not in other._ddf.columns:
            raise ValueError("right_on must be a column in the right catalog")

        ddf, ddf_map, alignment = join_catalog_data_on(self, other, left_on, right_on, suffixes=suffixes)

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
            catalog_name=output_catalog_name,
            ra_column=self.hc_structure.catalog_info.ra_column + suffixes[0],
            dec_column=self.hc_structure.catalog_info.dec_column + suffixes[0],
        )

        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf))
        return Catalog(ddf, ddf_map, hc_catalog)

    def join_nested(
        self,
        other: Catalog,
        left_on: str | None = None,
        right_on: str | None = None,
        nested_column_name: str | None = None,
        output_catalog_name: str | None = None,
    ) -> Catalog:
        """Perform a spatial join to another catalog by adding the other catalog as a nested column

        Joins two catalogs together on a shared column value, merging rows where they match.

        The result is added as a nested dataframe column using
        `nested-dask <https://github.com/lincc-frameworks/nested-dask>`__, where the right catalog's columns
        are encoded within a column in the resulting dataframe. For more information, view the
        `nested-dask documentation <https://nested-dask.readthedocs.io/en/latest/>`__.

        The operation only joins data from matching partitions and their margin caches, and does not join rows
        that have a matching column value but are in separate partitions in the sky. For a more general join,
        see the `merge` function.

        Args:
            other (Catalog): the right catalog to join to
            left_on (str): the name of the column in the left catalog to join on
            right_on (str): the name of the column in the right catalog to join on
            nested_column_name (str): the name of the nested column in the resulting dataframe storing the
                joined columns in the right catalog. (Default: name of right catalog)
            output_catalog_name (str): The name of the resulting catalog to be stored in metadata

        Returns:
            A new catalog with the columns from each of the input catalogs with their respective suffixes
            added, and the rows merged on the specified columns.
        """

        if left_on is None or right_on is None:
            raise ValueError("Both of left_on and right_on must be set")

        if left_on not in self._ddf.columns:
            raise ValueError("left_on must be a column in the left catalog")

        if right_on not in other._ddf.columns:
            raise ValueError("right_on must be a column in the right catalog")

        ddf, ddf_map, alignment = join_catalog_data_nested(
            self, other, left_on, right_on, nested_column_name=nested_column_name
        )

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(catalog_name=output_catalog_name)

        hc_catalog = hc.catalog.Catalog(new_catalog_info, alignment.pixel_tree)
        return Catalog(ddf, ddf_map, hc_catalog)

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = no_default,
        thresh: int | lib.NoDefault = no_default,
        on_nested: bool = False,
        subset: IndexLabel | None = None,
        ignore_index: bool = False,
    ) -> Catalog:
        catalog = super().dropna(
            axis=axis, how=how, thresh=thresh, on_nested=on_nested, subset=subset, ignore_index=ignore_index
        )
        if self.margin is not None:
            catalog.margin = self.margin.dropna(
                axis=axis,
                how=how,
                thresh=thresh,
                on_nested=on_nested,
                subset=subset,
                ignore_index=ignore_index,
            )
        return catalog
