from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, cast

import astropy
import dask
import dask.dataframe as dd
import hats as hc
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import BaseFrame
from dask.delayed import Delayed, delayed
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.inspection.visualize_catalog import get_fov_moc_from_wcs, initialize_wcs_axes, plot_healpix_map
from hats.pixel_math import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN
from matplotlib.figure import Figure
from pandas._libs import lib
from pandas._typing import AnyAll, Axis, IndexLabel
from pandas.api.extensions import no_default
from typing_extensions import Self
from upath import UPath

from lsdb import io
from lsdb.catalog.dataset.dataset import Dataset
from lsdb.core.plotting.plot_points import plot_points
from lsdb.core.plotting.skymap import compute_skymap, perform_inner_skymap
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.core.search.moc_search import MOCSearch
from lsdb.dask.merge_catalog_functions import concat_metas
from lsdb.io.schema import get_arrow_schema
from lsdb.types import DaskDFPixelMap


# pylint: disable=W0212
class HealpixDataset(Dataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hats.Dataset` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: HCHealpixDataset

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: HCHealpixDataset,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hats.Catalog` object with hats metadata of the catalog
        """
        super().__init__(ddf, hc_structure)
        self._ddf_pixel_map = ddf_pixel_map

    def __getitem__(self, item):
        result = self._ddf.__getitem__(item)
        if isinstance(result, nd.NestedFrame):
            return self.__class__(result, self._ddf_pixel_map, self.hc_structure)
        return result

    def __len__(self):
        """The number of rows in the catalog.

        Returns:
            The number of rows in the catalog, as specified in its metadata.
            This value is undetermined when the catalog is modified, and
            therefore an error is raised.
        """
        return len(self.hc_structure)

    def _create_modified_hc_structure(self, **kwargs) -> HCHealpixDataset:
        """Copy the catalog structure and override the specified catalog info parameters.

        Returns:
            A copy of the catalog's structure with updated info parameters.
        """
        return self.hc_structure.__class__(
            catalog_info=self.hc_structure.catalog_info.copy_and_update(**kwargs),
            pixels=self.hc_structure.pixel_tree,
            catalog_path=self.hc_structure.catalog_path,
            schema=self.hc_structure.schema,
            moc=self.hc_structure.moc,
        )

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

    def get_partition(self, order: int, pixel: int) -> nd.NestedFrame:
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
        if hp_pixel not in self._ddf_pixel_map:
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
        ndf = self._ddf.query(expr)
        hc_structure = self._create_modified_hc_structure(total_rows=0)
        return self.__class__(ndf, self._ddf_pixel_map, hc_structure)

    def _perform_search(
        self,
        metadata: hc.catalog.Catalog | hc.catalog.MarginCatalog,
        search: AbstractSearch,
    ) -> Tuple[DaskDFPixelMap, nd.NestedFrame]:
        """Performs a search on the catalog from a list of pixels to search in

        Args:
            metadata (hc.catalog.Catalog | hc.catalog.MarginCatalog): The metadata of
                the hats catalog after the coarse filtering is applied. The partitions
                it contains are only those that overlap with the spatial region.
            search (AbstractSearch): Instance of AbstractSearch.

        Returns:
            A tuple containing a dictionary mapping pixel to partition index and a dask dataframe
            containing the search results
        """
        filtered_pixels = metadata.get_healpix_pixels()
        if len(filtered_pixels) == 0:
            return {}, nd.NestedFrame.from_pandas(self._ddf._meta)
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
        return self.__class__(search_ndf, ddf_partition_map, filtered_hc_structure)

    def map_partitions(
        self,
        func: Callable[..., npd.NestedFrame],
        *args,
        meta: pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None = None,
        include_pixel: bool = False,
        **kwargs,
    ) -> Self | dd.Series:
        """Applies a function to each partition in the catalog.

        The ra and dec of each row is assumed to remain unchanged.

        Args:
            func (Callable): The function applied to each partition, which will be called with:
                `func(partition: npd.NestedFrame, *args, **kwargs)` with the additional args and kwargs passed
                to the `map_partitions` function. If the `include_pixel` parameter is set, the function will
                be called with the `healpix_pixel` as the second positional argument set to the healpix pixel
                of the partition as
                `func(partition: npd.NestedFrame, healpix_pixel: HealpixPixel, *args, **kwargs)`
            *args: Additional positional arguments to call `func` with.
            meta (pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None): An empty pandas DataFrame that
                has columns matching the output of the function applied to a partition. Other types are
                accepted to describe the output dataframe format, for full details see the dask documentation
                https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
                If meta is None (default), LSDB will try to work out the output schema of the function by
                calling the function with an empty DataFrame. If the function does not work with an empty
                DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
                will generate empty partitions, though these can be removed by calling the
                `Catalog.prune_empty_partitions` method.
            include_pixel (bool): Whether to pass the Healpix Pixel of the partition as a `HealpixPixel`
                object to the second positional argument of the function
            **kwargs: Additional keyword args to pass to the function. These are passed to the Dask DataFrame
                `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
                `transform_divisions` will be passed through and work as described in the dask documentation
                https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html

        Returns:
            A new catalog with each partition replaced with the output of the function applied to the original
            partition. If the function returns a non dataframe output, a dask Series will be returned.
        """
        if meta is None:
            if include_pixel:
                meta = func(self._ddf._meta.copy(), HealpixPixel(0, 0))
            else:
                meta = func(self._ddf._meta.copy())
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
                return func(df, pixels[partition_info["number"]], *args, **kwargs)

            output_ddf = self._ddf.map_partitions(apply_func, *args, meta=meta, **kwargs)
        else:
            output_ddf = self._ddf.map_partitions(func, *args, meta=meta, **kwargs)

        if isinstance(output_ddf, nd.NestedFrame) | isinstance(output_ddf, dd.DataFrame):
            return self.__class__(
                nd.NestedFrame.from_dask_dataframe(output_ddf), self._ddf_pixel_map, self.hc_structure
            )
        warnings.warn(
            "output of the function must be a DataFrame to generate an LSDB `Catalog`. `map_partitions` "
            "will return a dask object instead of a Catalog.",
            RuntimeWarning,
        )
        return output_ddf

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
        search_ddf = (
            self._ddf.partitions[non_empty_partitions]
            if len(non_empty_partitions) > 0
            else nd.NestedFrame.from_pandas(self._ddf._meta, npartitions=1)
        )
        ddf_partition_map = {pixel: i for i, pixel in enumerate(non_empty_pixels)}
        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(non_empty_pixels)
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)

    def _get_non_empty_partitions(self) -> Tuple[List[HealpixPixel], np.ndarray]:
        """Determines which pixels and partitions of a catalog are not empty

        Returns:
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

    def skymap_data(
        self,
        func: Callable[[npd.NestedFrame, HealpixPixel], Any],
        order: int | None = None,
        default_value: Any = 0.0,
        **kwargs,
    ) -> Dict[HealpixPixel, Delayed]:
        """Perform a function on each partition of the catalog, returning a dict of values for each pixel.

        Args:
            func (Callable[[npd.NestedFrame, HealpixPixel], Any]): A function that takes a pandas
                DataFrame with the data in a partition, the HealpixPixel of the partition, and any other
                keyword arguments and returns an aggregated value
            order (int | None): The HEALPix order to compute the skymap at. If None (default),
                will compute for each partition in the catalog at their own orders. If a value
                other than None, each partition will be grouped by pixel number at the order
                specified and the function will be applied to each group.
            default_value (Any): The value to use at pixels that aren't covered by the catalog (default 0)
            **kwargs: Arguments to pass to the function

        Returns:
            A dict of Delayed values, one for the function applied to each partition of the catalog.
            If order is not None, the Delayed objects will be numpy arrays with all pixels within the
            partition at the specified order. Any pixels within a partition that have no coverage will
            have the default_value as its result, as well as any pixels for which the aggregate
            function returns None.
        """
        results = {}
        partitions = self.to_delayed()
        if order is None:
            results = {
                pixel: delayed(func)(partitions[index], pixel, **kwargs)
                for pixel, index in self._ddf_pixel_map.items()
            }
        elif len(self.hc_structure.pixel_tree) > 0:
            if order < self.hc_structure.pixel_tree.get_max_depth():
                raise ValueError(
                    f"order must be greater than or equal to max order in catalog "
                    f"({self.hc_structure.pixel_tree.get_max_depth()})"
                )
            results = {
                pixel: perform_inner_skymap(partitions[index], func, pixel, order, default_value, **kwargs)
                for pixel, index in self._ddf_pixel_map.items()
            }
        return results

    def skymap_histogram(
        self,
        func: Callable[[npd.NestedFrame, HealpixPixel], Any],
        order: int | None = None,
        default_value: Any = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Get a histogram with the result of a given function applied to the points in each HEALPix pixel of
        a given order

        Args:
            func (Callable[[npd.NestedFrame, HealpixPixel], Any]): A function that takes a pandas DataFrame
                and the HealpixPixel the partition is from and returns a value
            order (int | None): The HEALPix order to compute the skymap at. If None (default),
                will compute for each partition in the catalog at their own orders. If a value
                other than None, each partition will be grouped by pixel number at the order
                specified and the function will be applied to each group.
            default_value (Any): The value to use at pixels that aren't covered by the catalog (default 0)
            **kwargs: Arguments to pass to the given function

        Returns:
            A 1-dimensional numpy array where each index i is equal to the value of the function applied to
            the points within the HEALPix pixel with pixel number i in NESTED ordering at a specified order.
            If no order is supplied, the order of the resulting histogram will be the highest order partition
            in the catalog, and the function will be applied to the partitions of the catalog with the result
            copied to all pixels if the catalog partition is at a lower order than the histogram order.

            If order is specified, any pixels at the specified order not covered by the catalog or any pixels
            that the function returns None will use the default_value.
        """
        smdata = self.skymap_data(func, order, default_value, **kwargs)
        pixels = list(smdata.keys())
        results = dask.compute(*[smdata[pixel] for pixel in pixels])
        result_dict = {pixels[i]: results[i] for i in range(len(pixels))}
        return compute_skymap(result_dict, order, default_value)

    def skymap(
        self,
        func: Callable[[npd.NestedFrame, HealpixPixel], Any],
        order: int | None = None,
        default_value: Any = 0,
        projection="MOL",
        plotting_args: Dict | None = None,
        **kwargs,
    ) -> tuple[Figure, WCSAxes]:
        """Plot a skymap of an aggregate function applied over each partition

        Args:
            func (Callable[[npd.NestedFrame, HealpixPixel], Any]): A function that takes a pandas DataFrame
                and the HealpixPixel the partition is from and returns a value
            order (int | None): The HEALPix order to compute the skymap at. If None (default),
                will compute for each partition in the catalog at their own orders. If a value
                other than None, each partition will be grouped by pixel number at the order
                specified and the function will be applied to each group.
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
        if plotting_args is None:
            plotting_args = {}
        return plot_healpix_map(img, projection=projection, **plotting_args)

    def plot_pixels(self, projection: str = "MOL", **kwargs) -> tuple[Figure, WCSAxes]:
        """Create a visual map of the pixel density of the catalog.

        Args:
            projection (str) The map projection to use. Available projections listed at
            https://docs.astropy.org/en/stable/wcs/supported_projections.html
            kwargs (dict): additional keyword arguments to pass to plotting call.
        """
        return self.hc_structure.plot_pixels(projection=projection, **kwargs)

    def plot_coverage(self, **kwargs) -> tuple[Figure, WCSAxes]:
        """Create a visual map of the coverage of the catalog.

        Args:
            kwargs: additional keyword arguments to pass to hats.Catalog.plot_moc
        """
        return self.hc_structure.plot_moc(**kwargs)

    def to_hats(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Saves the catalog to disk in HATS format

        Args:
            base_catalog_path (str): Location where catalog is saved to
            catalog_name (str): The name of the catalog to be saved
            overwrite (bool): If True existing catalog is overwritten
            **kwargs: Arguments to pass to the parquet write operations
        """
        default_histogram_order = 8
        max_catalog_depth = self.hc_structure.pixel_tree.get_max_depth()
        histogram_order = max(max_catalog_depth, default_histogram_order)
        io.to_hats(
            self,
            base_catalog_path=base_catalog_path,
            catalog_name=catalog_name,
            histogram_order=histogram_order,
            overwrite=overwrite,
            **kwargs,
        )

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll | lib.NoDefault = no_default,
        thresh: int | lib.NoDefault = no_default,
        on_nested: bool = False,
        subset: IndexLabel | None = None,
        ignore_index: bool = False,
    ) -> Self:  # type: ignore[name-defined] # noqa: F821:
        """
        Remove missing values for one layer of nested columns in the catalog.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
            * 1, or 'columns' : Drop columns which contain missing value.

            Only a single axis is allowed.

        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from catalog, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.
        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        on_nested : str or bool, optional
            If not False, applies the call to the nested dataframe in the
            column with label equal to the provided string. If specified,
            the nested dataframe should align with any columns given in
            `subset`.
        subset : column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.

            Access nested columns using `nested_df.nested_col` (where
            `nested_df` refers to a particular nested dataframe and
            `nested_col` is a column of that nested dataframe).
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Catalog
            Catalog with NA entries dropped from it.

        Notes
        -----
        Operations that target a particular nested structure return a dataframe
        with rows of that particular nested structure affected.

        Values for `on_nested` and `subset` should be consistent in pointing
        to a single layer, multi-layer operations are not supported at this
        time.
        """

        def drop_na_part(df: npd.NestedFrame):
            if df.index.name == SPATIAL_INDEX_COLUMN:
                df = df.reset_index()
            df = cast(
                npd.NestedFrame,
                df.dropna(
                    axis=axis,
                    how=how,
                    thresh=thresh,
                    on_nested=on_nested,
                    subset=subset,
                    ignore_index=ignore_index,
                ),
            )
            if SPATIAL_INDEX_COLUMN in df.columns:
                df = df.set_index(SPATIAL_INDEX_COLUMN)
            return df

        ndf = self._ddf.map_partitions(drop_na_part, meta=self._ddf._meta)
        hc_structure = self._create_modified_hc_structure(total_rows=0)
        return self.__class__(ndf, self._ddf_pixel_map, hc_structure)

    def nest_lists(
        self,
        base_columns: list[str] | None,
        list_columns: list[str] | None = None,
        name: str = "nested",
    ) -> Self:  # type: ignore[name-defined] # noqa: F821:
        """Creates a new catalog with a set of list columns packed into a
        nested column.

        Args:
            base_columns (list-like or None): Any columns that have non-list values in the input catalog.
            These will simply be kept as identical columns in the result
        list_columns (list-like or None): The list-value columns that should be packed into a nested column.
            All columns in the list will attempt to be packed into a single
                nested column with the name provided in `nested_name`. All columns
                in list_columns must have pyarrow list dtypes, otherwise the
                operation will fail. If None, is defined as all columns not in
                `base_columns`.
        name (str): The name of the output column the `nested_columns` are packed into.

        Returns:
            A new catalog with specified list columns nested into a new nested column.

        Note:
            As noted above, all columns in `list_columns` must have a pyarrow
            ListType dtype. This is needed for proper meta propagation. To convert
            a list column to this dtype, you can use this command structure:
            `nf= nf.astype({"colname": pd.ArrowDtype(pa.list_(pa.int64()))})`
            Where pa.int64 above should be replaced with the correct dtype of the
            underlying data accordingly.
            Additionally, it's a known issue in Dask
            (https://github.com/dask/dask/issues/10139) that columns with list
            values will by default be converted to the string type. This will
            interfere with the ability to recast these to pyarrow lists. We
            recommend setting the following dask config setting to prevent this:
            `dask.config.set({"dataframe.convert-string":False})`
        """
        new_ddf = nd.NestedFrame.from_lists(
            self._ddf,
            base_columns=base_columns,
            list_columns=list_columns,
            name=name,
        )
        hc_structure = self._create_modified_hc_structure(total_rows=0)
        return self.__class__(new_ddf, self._ddf_pixel_map, hc_structure)

    def reduce(self, func, *args, meta=None, append_columns=False, **kwargs) -> Self:
        """
        Takes a function and applies it to each top-level row of the Catalog.

        docstring copied from nested-pandas

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passsed to the function as
        scalars and columns from the nested layers being passed as numpy arrays.

        Parameters
        ----------
        func : callable
            Function to apply to each nested dataframe. The first arguments to `func` should be which
            columns to apply the function to. See the Notes for recommendations
            on writing func outputs.
        args : positional arguments
            Positional arguments to pass to the function, the first *args should be the names of the
            columns to apply the function to.
        meta : dataframe or series-like, optional
            The dask meta of the output. If append_columns is True, the meta should specify just the
            additional columns output by func.
        append_columns : bool
            If the output columns should be appended to the orignal dataframe.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `HealpixDataset`
            `HealpixDataset` with the results of the function applied to the columns of the frame.

        Notes
        -----
        By default, `reduce` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. For more useful
        naming, it's recommended to have `func` return a dictionary where each
        key is an output column of the dataframe returned by `reduce`.

        Example User Function:

        >>> def my_sum(col1, col2):
        >>>    '''reduce will return a NestedFrame with two columns'''
        >>>    return {"sum_col1": sum(col1), "sum_col2": sum(col2)}
        >>>
        >>> catalog.reduce(my_sum, 'sources.col1', 'sources.col2')

        """

        if append_columns:
            meta = concat_metas([self._ddf._meta.copy(), meta])

        catalog_info = self.hc_structure.catalog_info

        def reduce_part(df):
            reduced_result = npd.NestedFrame(df).reduce(func, *args, **kwargs)
            if append_columns:
                if catalog_info.ra_column in reduced_result or catalog_info.dec_column in reduced_result:
                    raise ValueError("ra and dec columns can not be modified using reduce")
                return npd.NestedFrame(pd.concat([df, reduced_result], axis=1))
            return reduced_result

        ndf = nd.NestedFrame.from_dask_dataframe(self._ddf.map_partitions(reduce_part, meta=meta))

        hc_updates: dict = {"total_rows": 0}
        if not append_columns:
            hc_updates = {**hc_updates, "ra_column": "", "dec_column": ""}

        hc_catalog = self._create_modified_hc_structure(**hc_updates)
        hc_catalog.schema = get_arrow_schema(ndf)
        return self.__class__(ndf, self._ddf_pixel_map, hc_catalog)

    def plot_points(
        self,
        *,
        ra_column: str | None = None,
        dec_column: str | None = None,
        color_col: str | None = None,
        projection: str = "MOL",
        title: str | None = None,
        fov: Quantity | Tuple[Quantity, Quantity] | None = None,
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
        If the fov or wcs args are set, only the partitions in the catalog visible to the plot will be
        computed.
        The scatter points can be colored by a column of the catalog by using the `color_col` kwarg

        Args:
            ra_column (str | None): The column to use as the RA of the points to plot. Defaults to the
                catalog's default RA column. Useful for plotting joined or cross-matched points
            dec_column (str | None): The column to use as the Declination of the points to plot. Defaults to
                the catalog's default Declination column. Useful for plotting joined or cross-matched points
            color_col (str | None): The column to use as the color array for the scatter plot. Allows coloring
                of the points by the values of a given column.
            projection (str): The projection to use in the WCS. Available projections listed at
                https://docs.astropy.org/en/stable/wcs/supported_projections.html
            title (str): The title of the plot
            fov (Quantity or Sequence[Quantity, Quantity] | None): The Field of View of the WCS. Must be an
                astropy Quantity with an angular unit, or a tuple of quantities for different longitude and \
                latitude FOVs (Default covers the full sky)
            center (SkyCoord | None): The center of the projection in the WCS (Default: SkyCoord(0, 0))
            wcs (WCS | None): The WCS to specify the projection of the plot. If used, all other WCS parameters
                are ignored and the parameters from the WCS object is used.
            frame_class (Type[BaseFrame] | None): The class of the frame for the WCSAxes to be initialized
                with. if the `ax` kwarg is used, this value is ignored (By Default uses EllipticalFrame for
                full sky projection. If FOV is set, RectangularFrame is used)
            ax (WCSAxes | None): The matplotlib axes to plot onto. If None, an axes will be created to be
                used. If specified, the axes must be an astropy WCSAxes, and the `wcs` parameter must be set
                with the WCS object used in the axes. (Default: None)
            fig (Figure | None): The matplotlib figure to add the axes to. If None, one will be created,
                unless ax is specified (Default: None)
            **kwargs: Additional kwargs to pass to creating the matplotlib `scatter` function. These include
                `c` for color, `s` for the size of hte points, `marker` for the maker type, `cmap` and `norm`
                if `color_col` is used

        Returns:
            Tuple[Figure, WCSAxes] - The figure and axes used for the plot
        """
        fig, ax, wcs = initialize_wcs_axes(
            projection=projection,
            fov=fov,
            center=center,
            wcs=wcs,
            frame_class=frame_class,
            ax=ax,
            fig=fig,
            figsize=(9, 5),
        )

        fov_moc = get_fov_moc_from_wcs(wcs)

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
