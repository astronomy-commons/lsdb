from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Type

import dask.dataframe as dd
import hats as hc
import nested_pandas as npd
import pandas as pd
from hats.catalog.catalog_collection import CatalogCollection
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.catalog.index.index_catalog import IndexCatalog as HCIndexCatalog
from pandas._typing import Renamer
from typing_extensions import Self
from upath import UPath

from lsdb import io
from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.catalog.map_catalog import MapCatalog
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.core.search.index_search import IndexSearch
from lsdb.dask.concat_catalog_data import concat_catalog_data, concat_margin_data
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data, crossmatch_catalog_data_nested
from lsdb.dask.join_catalog_data import (
    join_catalog_data_nested,
    join_catalog_data_on,
    join_catalog_data_through,
    merge_asof_catalog_data,
)
from lsdb.dask.merge_catalog_functions import create_merged_catalog_info
from lsdb.dask.merge_map_catalog_data import merge_map_catalog_data
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.nested.core import NestedFrame
from lsdb.types import DaskDFPixelMap


# pylint: disable=protected-access,too-many-public-methods, too-many-lines
class Catalog(HealpixDataset):
    """LSDB Catalog DataFrame to perform analysis of sky catalogs and efficient
    spatial operations.
    """

    hc_structure: hc.catalog.Catalog
    """`hats.Catalog` object representing (only) the structure and metadata of the HATS catalog"""

    margin: MarginCatalog | None = None
    """Link to a ``MarginCatalog`` object that represents the objects in other partitions that
    are within a specified radius of the border with this partition. This is useful for 
    finding best counterparts when crossmatching catalogs."""

    hc_collection: CatalogCollection | None = None
    """`hats.CatalogCollection` object representing the structure and metadata of the
    HATS catalog, as well as links to affiliated tables like margins and indexes."""

    def __init__(
        self,
        ddf: NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.Catalog,
        *,
        loading_config: HatsLoadingConfig | None = None,
        margin: MarginCatalog | None = None,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.open_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hats.Catalog` object with hats metadata of the catalog
        """
        super().__init__(ddf, ddf_pixel_map, hc_structure, loading_config)
        self.margin = margin

    def _create_updated_dataset(
        self,
        ddf: NestedFrame | None = None,
        ddf_pixel_map: DaskDFPixelMap | None = None,
        hc_structure: HCHealpixDataset | None = None,
        updated_catalog_info_params: dict | None = None,
        margin: MarginCatalog | None = None,
    ) -> Self:
        cat = super()._create_updated_dataset(
            ddf,
            ddf_pixel_map,
            hc_structure,
            updated_catalog_info_params,
        )
        cat.margin = margin
        return cat

    @property
    def iloc(self):
        """Returns the position-indexer for the catalog"""
        raise NotImplementedError(
            "Access via .iloc is not supported since it would require computing the entire catalog."
        )

    @property
    def loc(self):
        """Returns the label-indexer for the catalog"""
        raise NotImplementedError(
            "Access via .loc is not allowed. Please use `Catalog.id_search` instead."
            " For example, to retrieve a row for an object of ID 'GAIA_123' use"
            " catalog.id_search(values={'objid':'GAIA_123'}), where 'objid' is the"
            " column for which there is an index catalog. If `id_search` is targeted"
            " at a column other than the collection's default index column, or if"
            " working with a stand-alone catalog, use the `index_catalogs` argument"
            " to specify a HATS index catalog for the desired column."
        )

    def query(self, expr: str) -> Catalog:
        """Filters catalog and respective margin, if it exists, using a complex query expression

        Args:
            expr (str): Query expression to evaluate. The column names that are not valid Python
                variables names should be wrapped in backticks, and any variable values can be
                injected using f-strings. The use of '@' to reference variables is not supported.
                More information about pandas query strings is available
                `here <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`__.

        Returns:
            A catalog that contains the data from the original catalog that complies with the query
            expression. If a margin exists, it is filtered according to the same query expression.
        """
        catalog = super().query(expr)
        if self.margin is not None:
            catalog.margin = self.margin.query(expr)
        return catalog

    def rename(self, columns: Renamer) -> Catalog:
        """Renames catalog columns (not indices) and that of its margin if it exists using a
        dictionary or function mapping.

        Args:
            columns (dict-like or function): transformations to apply to column names.

        Returns:
            A catalog that contains the data from the original catalog with renamed columns.
            If a margin exists, it is renamed according to the same column name mapping.
        """
        catalog = super().rename(columns)
        if self.margin is not None:
            catalog.margin = self.margin.rename(columns)
        return catalog

    def crossmatch(
        self,
        other: Catalog,
        suffixes: tuple[str, str] | None = None,
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
        if not isinstance(other, Catalog):
            raise TypeError(
                f"Expected `other` to be a Catalog instance, got {type(other)}. "
                "You may want `lsdb.crossmatch(frame_or_catalog, frame_or_catalog)` instead."
            )

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
        new_catalog_info = create_merged_catalog_info(
            self.hc_structure.catalog_info, other.hc_structure.catalog_info, output_catalog_name, suffixes
        )
        hc_catalog = self.hc_structure.__class__(
            new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf), moc=alignment.moc
        )
        return self.__class__(ddf, ddf_map, hc_catalog)

    def crossmatch_nested(
        self,
        other: Catalog,
        nested_column_name: str | None = None,
        algorithm: (
            Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
        ) = BuiltInCrossmatchAlgorithm.KD_TREE,
        output_catalog_name: str | None = None,
        require_right_margin: bool = False,
        **kwargs,
    ) -> Catalog:
        """Perform a cross-match between two catalogs, adding the result as a nested column

        For each row in the left catalog, the cross-matched rows from the right catalog are added
        in a new nested column. Any extra columns from the crossmatch like distance are added to this
        nested column too.

        The pixels from each catalog are aligned via a `PixelAlignment`, and cross-matching is
        performed on each pair of overlapping pixels. The resulting catalog will have partitions
        matching an inner pixel alignment - using pixels that have overlap in both input catalogs
        and taking the smallest of any overlapping pixels.

        The resulting catalog will be partitioned using the left catalog's ra and dec, and the
        index for each row will be the same as the index from the corresponding row in the left
        catalog's index.

        Args:
            other (Catalog): The right catalog to cross-match against
            nested_column_name (str): The name of the nested column that will contain the crossmatched rows
                from the right catalog. Default: uses the name of the right catalog
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

                    You may add any additional keyword argument parameters to the crossmatch
                    function definition, and the user will be able to pass them in as kwargs in the
                    `Catalog.crossmatch` method. Any additional keyword arguments must also be added to the
                    `CrossmatchAlgorithm.validate` classmethod by overwriting the method.

            output_catalog_name (str): The name of the resulting catalog.
                Default: {left_name}_x_{right_name}
            require_right_margin (bool): If true, raises an error if the right margin is missing which could
                lead to incomplete crossmatches. Default: False

        Returns:
            A Catalog with the data from the left and right catalogs joined with the cross-matched rows from
            the right catalog added in a new nested column.

            The resulting table contains all columns from the left catalog and a new nested column with all
            the columns from the right catalog and any extra columns generated by the crossmatch algorithm.
        """
        if nested_column_name is None:
            nested_column_name = other.name
        if other.margin is None and require_right_margin:
            raise ValueError("Right catalog margin cache is required for cross-match.")
        if output_catalog_name is None:
            output_catalog_name = f"{self.name}_x_{other.name}"
        ddf, ddf_map, alignment = crossmatch_catalog_data_nested(
            self, other, nested_column_name, algorithm=algorithm, **kwargs
        )
        hc_catalog = self.hc_structure.__class__(
            self.hc_structure.catalog_info,
            alignment.pixel_tree,
            moc=alignment.moc,
        )
        return self._create_updated_dataset(
            ddf=ddf,
            ddf_pixel_map=ddf_map,
            hc_structure=hc_catalog,
            updated_catalog_info_params={"catalog_name": output_catalog_name},
        )

    def concat(
        self,
        other: Catalog,
        **kwargs,
    ) -> Catalog:
        """
        Concatenate two catalogs by aligned HEALPix pixels.

        Args:
            other (Catalog): The catalog to concatenate with.
            **kwargs: Extra arguments forwarded to internal `pandas.concat` calls.

        Returns:
            Catalog: A new catalog whose partitions correspond to the OUTER pixel alignment
            and whose rows are the per-pixel concatenation of both inputs. If both
            inputs provide a margin, the result includes a concatenated margin
            dataset as described above.

        Raises:
            Warning: If only one side has a margin, a warning is emitted and the result will
            not include a margin dataset.

        Notes:
            - The main (non-margin) alignment is filtered by the catalogs’ MOCs when
              available; the pixel-tree alignment itself is OUTER, so pixels present on
              either side are preserved (within the MOC filter).
            - This is a stacking operation, not a row-wise join or crossmatch; no
              deduplication or key-based matching is applied.
            - Column dtypes may be upcast by pandas to accommodate the unioned schema.
              Row/column order is not guaranteed to be stable.
            - `**kwargs` are forwarded to the internal pandas concatenations (e.g.,
              `ignore_index`, etc.).
        """
        # check if the catalogs have margins
        margin = None
        if self.margin is None and other.margin is not None:
            warnings.warn(
                "Left catalog has no margin, result will not include margin data.",
            )

        if self.margin is not None and other.margin is None:
            warnings.warn(
                "Right catalog has no margin, result will not include margin data.",
            )

        if self.margin is not None and other.margin is not None:
            smallest_margin_radius = min(
                self.margin.hc_structure.catalog_info.margin_threshold or 0,
                other.margin.hc_structure.catalog_info.margin_threshold or 0,
            )

            margin_ddf, margin_ddf_map, margin_alignment = concat_margin_data(
                self, other, smallest_margin_radius, **kwargs
            )
            margin_hc_catalog = self.margin.hc_structure.__class__(
                self.margin.hc_structure.catalog_info,
                margin_alignment.pixel_tree,
            )
            margin = self.margin._create_updated_dataset(
                ddf=margin_ddf,
                ddf_pixel_map=margin_ddf_map,
                hc_structure=margin_hc_catalog,
                updated_catalog_info_params={"margin_threshold": smallest_margin_radius},
            )

        ddf, ddf_map, alignment = concat_catalog_data(self, other, **kwargs)
        hc_catalog = self.hc_structure.__class__(
            self.hc_structure.catalog_info,
            alignment.pixel_tree,
            moc=alignment.moc,
        )
        return self._create_updated_dataset(
            ddf=ddf,
            ddf_pixel_map=ddf_map,
            hc_structure=hc_catalog,
            margin=margin,
        )

    def merge_map(
        self,
        map_catalog: MapCatalog,
        func: Callable[..., npd.NestedFrame],
        *args,
        meta: npd.NestedFrame | None = None,
        **kwargs,
    ) -> Catalog:
        """Applies a function to each pair of partitions in this catalog and the map catalog.

        The pixels from each catalog are aligned via a `PixelAlignment`, and the respective dataframes
        are passed to the function. The resulting catalog will have the same partitions as the point
        source catalog.

        Args:
            map_catalog (MapCatalog): The continuous map to merge.
            func (Callable): The function applied to each catalog partition, which will be called with:
                `func(catalog_partition: npd.NestedFrame, map_partition: npd.NestedFrame, `
                ` healpix_pixel: HealpixPixel, *args, **kwargs)`
                with the additional args and kwargs passed to the `merge_map` function.
            *args: Additional positional arguments to call `func` with.
            meta (pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None): An empty pandas DataFrame that
                has columns matching the output of the function applied to the catalog partition. Other types
                are accepted to describe the output dataframe format, for full details see the dask
                documentation https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
                If meta is None (default), LSDB will try to work out the output schema of the function by
                calling the function with an empty DataFrame. If the function does not work with an empty
                DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
                will generate empty partitions, though these can be removed by calling the
                `Catalog.prune_empty_partitions` method.
            **kwargs: Additional keyword args to pass to the function. These are passed to the Dask DataFrame
                `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
                `transform_divisions` will be passed through and work as described in the dask documentation
                https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html

        Returns:
            A Catalog with the data from the left and right catalogs merged with one row for each
            pair of neighbors found from cross-matching.

            The resulting table contains all columns from the left and right catalogs with their
            respective suffixes and, whenever specified, a set of extra columns generated by the
            crossmatch algorithm.
        """
        ddf, ddf_map, alignment = merge_map_catalog_data(self, map_catalog, func, *args, meta=meta, **kwargs)
        hc_catalog = self.hc_structure.__class__(
            self.hc_structure.catalog_info,
            alignment.pixel_tree,
            schema=get_arrow_schema(ddf),
            moc=alignment.moc,
        )
        return self._create_updated_dataset(ddf=ddf, ddf_pixel_map=ddf_map, hc_structure=hc_catalog)

    def id_search(
        self,
        values: dict[str, Any],
        index_catalogs: dict[str, str | HCIndexCatalog] | None = None,
        fine: bool = True,
    ) -> Catalog:
        """Query rows by column values.

        In the context of Catalog collections this method will try to find an index
        catalog for each field name specified in `values`. If the catalog calling
        this method is not part of a collection or if it cannot find the index catalog
        for the fields in the collection specification, explicit `index_catalogs` for
        the desired fields can be specified. The `index_catalogs` argument is a dictionary
        of field names to HATS index catalog paths or their instances and they take
        precedence over the catalogs specified in the collection.

        Args:
            values (dict[str, Any]): The mapping of field names (as string) to their search values.
                Values can be single values or lists of values (but only one list of values is
                allowed per search). If a list is specified, the search will return rows that match
                any of the values in the list.
            index_catalogs (dict[str, str|HCIndexCatalog]): The mapping of field names (as string)
                to their respective index catalog paths or instance of `HCIndexCatalog`. Use this
                argument to specify index catalogs for stand-alone catalogs or for collections where
                there is no index catalog for the fields you are querying for.
            fine (bool): If True, the rows of the partitions where a column match occurred are
                filtered. If False, all the rows of those partitions are kept. Defaults to True.

        Example:
            To query by "objid" where an index for this field is available in the collection::

                catalog.id_search(values={"objid":"GAIA_123"})

            To query by "fieldid" and "ccid", if "fieldid" has an index catalog in the collection
            and the index catalog for "ccid" is present in a directory named "ccid_id_index_catalog"
            on the current working directory::

                catalog.id_search(
                    values={"fieldid": 700, "ccid": 300},
                    index_catalogs={"ccid": "ccid_id_index_catalog"}
                )

            To query for multiple values in a column, use a list of values::

                catalog.id_search(values={"objid": [1, 2, 3, ...]})

        Returns:
            A new Catalog containing the results of the column match.
        """
        # Only one column may contain a list of values (multiple columns may be specified, so long
        # as only one of them is a list).
        if not values:
            raise ValueError("No values specified for search.")
        list_value_already_found = False
        for field, value in values.items():
            print(f"Searching for {field}={value}")
            if isinstance(value, list):
                if list_value_already_found:
                    raise ValueError("Only one column may contain a list of values.")
                list_value_already_found = True

        self._check_unloaded_columns(list(values.keys()))

        def _get_index_catalog_for_field(field: str):
            """Find the index catalog for `field`. Index catalogs declared
            as an argument take precedence over the ones in the collection"""
            field_index: str | Path | UPath | HCIndexCatalog | None = None
            if index_catalogs is not None and field in index_catalogs:
                field_index = index_catalogs[field]
            elif self.hc_collection is not None:
                field_index = self.hc_collection.get_index_dir_for_field(field)
            if isinstance(field_index, HCIndexCatalog):
                return field_index
            if isinstance(field_index, (str | Path | UPath)):
                return hc.read_hats(field_index)
            raise TypeError(f"Catalog index for field `{field}` is not of type `HCIndexCatalog`")

        field_indexes = {field_name: _get_index_catalog_for_field(field_name) for field_name in values.keys()}
        return self.search(IndexSearch(values, field_indexes, fine))

    def search(self, search: AbstractSearch):
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria.
        Filters to points that match some finer criteria.

        Args:
            search (AbstractSearch): Instance of AbstractSearch.

        Returns:
            A new Catalog containing the points filtered to those matching the search parameters.
        """
        cat = super().search(search)
        cat.margin = self.margin.search(search) if self.margin is not None else None
        return cat

    def map_partitions(
        self,
        func: Callable[..., npd.NestedFrame],
        *args,
        meta: pd.DataFrame | pd.Series | dict | Iterable | tuple | None = None,
        include_pixel: bool = False,
        **kwargs,
    ) -> Catalog | dd.Series:
        """Applies a function to each partition in the catalog and respective margin.

        The ra and dec of each row is assumed to remain unchanged. If the function returns a DataFrame,
        an LSDB Catalog is constructed and its respective margin is updated accordingly, if it exists.
        Otherwise, only the main catalog Dask object is returned.

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
        catalog = super().map_partitions(
            func,
            *args,
            meta=meta,
            include_pixel=include_pixel,
            **kwargs,
        )
        if isinstance(catalog, Catalog) and self.margin is not None:
            catalog.margin = self.margin.map_partitions(
                func,
                *args,
                meta=meta,
                include_pixel=include_pixel,
                **kwargs,
            )  # type: ignore[assignment]
        return catalog

    def merge(
        self,
        other: Catalog,
        how: str = "inner",
        on: str | list | None = None,
        left_on: str | list | None = None,
        right_on: str | list | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: tuple[str, str] | None = None,
    ) -> NestedFrame:
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

        def make_strlist(col: str | list[str] | None) -> list[str]:
            if col is None:
                return []
            if isinstance(col, str):
                return [col]
            return col

        names_to_check = make_strlist(on)
        if not left_index:
            names_to_check += make_strlist(left_on)
        if not right_index:
            names_to_check += make_strlist(right_on)
        self._check_unloaded_columns(names_to_check)
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
        suffixes: tuple[str, str] | None = None,
        output_catalog_name: str | None = None,
    ):
        """Uses the pandas `merge_asof` function to merge two catalogs on their indices by distance of keys

        Must be along catalog indices, and does not include margin caches, meaning results may be incomplete
        for merging points.

        This function is intended for use in special cases such as Dust Map Catalogs, for general merges,
        the ``crossmatch`` and ``join`` functions should be used.

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

        new_catalog_info = create_merged_catalog_info(
            self.hc_structure.catalog_info, other.hc_structure.catalog_info, output_catalog_name, suffixes
        )
        hc_catalog = hc.catalog.Catalog(
            new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf), moc=alignment.moc
        )
        return Catalog(ddf, ddf_map, hc_catalog)

    def join(
        self,
        other: Catalog,
        left_on: str | None = None,
        right_on: str | None = None,
        through: AssociationCatalog | None = None,
        suffixes: tuple[str, str] | None = None,
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

        self._check_unloaded_columns([left_on, right_on])

        if through is not None:
            ddf, ddf_map, alignment = join_catalog_data_through(self, other, through, suffixes)
        else:
            if left_on is None or right_on is None:
                raise ValueError("Either both of left_on and right_on, or through must be set")
            if left_on not in self._ddf.columns:
                raise ValueError("left_on must be a column in the left catalog")
            if right_on not in other._ddf.columns:
                raise ValueError("right_on must be a column in the right catalog")
            ddf, ddf_map, alignment = join_catalog_data_on(self, other, left_on, right_on, suffixes)

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = create_merged_catalog_info(
            self.hc_structure.catalog_info, other.hc_structure.catalog_info, output_catalog_name, suffixes
        )
        hc_catalog = hc.catalog.Catalog(
            new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf), moc=alignment.moc
        )
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

        self._check_unloaded_columns([left_on])
        if left_on not in self._ddf.columns:
            raise ValueError("left_on must be a column in the left catalog")

        other._check_unloaded_columns([right_on])
        if right_on not in other._ddf.columns:
            raise ValueError("right_on must be a column in the right catalog")

        ddf, ddf_map, alignment = join_catalog_data_nested(
            self, other, left_on, right_on, nested_column_name=nested_column_name
        )

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
            catalog_name=output_catalog_name, total_rows=0
        )
        hc_catalog = hc.catalog.Catalog(
            new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf), moc=alignment.moc
        )
        return Catalog(ddf, ddf_map, hc_catalog)

    def nest_lists(
        self,
        base_columns: list[str] | None = None,
        list_columns: list[str] | None = None,
        name: str = "nested",
    ) -> Catalog:
        """Creates a new catalog with a set of list columns packed into a nested column.

        Args:
            base_columns (list-like or None): Any columns that have non-list values in the input catalog.
                These will simply be kept as identical columns in the result. If None, is inferred to be
                all columns in the input catalog that are not considered list-value columns.
            list_columns (list-like or None): The list-value columns that should be packed into a nested
                column. All columns in the list will attempt to be packed into a single nested column with
                the name provided in ``nested_name``. All columns in list_columns must have pyarrow list
                dtypes, otherwise the operation will fail. If None, is defined as all columns not in
                ``base_columns``.
            name (str): The name of the output column the `nested_columns` are packed into.

        Returns:
            A new catalog with specified list columns nested into a new nested column.

        Note:
            As noted above, all columns in `list_columns` must have a pyarrow
            ListType dtype. This is needed for proper meta propagation. To convert
            a list column to this dtype, you can use this command structure:

                nf= nf.astype({"colname": pd.ArrowDtype(pa.list_(pa.int64()))})

            Where pa.int64 above should be replaced with the correct dtype of the
            underlying data accordingly.
            Additionally, it's a known issue in Dask
            (https://github.com/dask/dask/issues/10139) that columns with list
            values will by default be converted to the string type. This will
            interfere with the ability to recast these to pyarrow lists. We
            recommend setting the following dask config setting to prevent this:

                `dask.config.set({"dataframe.convert-string":False})`
        """
        catalog = super().nest_lists(
            base_columns=base_columns,
            list_columns=list_columns,
            name=name,
        )
        if self.margin is not None:
            catalog.margin = self.margin.nest_lists(
                base_columns=base_columns,
                list_columns=list_columns,
                name=name,
            )
        return catalog

    def reduce(self, func, *args, meta=None, append_columns=False, infer_nesting=True, **kwargs) -> Catalog:
        """
        Takes a function and applies it to each top-level row of the Catalog.

        docstring copied from nested-pandas

        The user may specify which columns the function is applied to, with
        columns from the 'base' layer being passsed to the function as
        scalars and columns from the nested layers being passed as numpy arrays.

        Parameters
        ----------
        func : callable
            Function to apply to each row in the catalog. The first arguments to `func` should be which
            columns to apply the function to. See the Notes for recommendations on writing func outputs.
        args : positional arguments
            A list of string column names to pull from the NestedFrame to pass along to the function.
            If the function has additional arguments, pass them as keyword arguments (e.g. arg_name=value)
        meta : dataframe or series-like, optional
            The dask meta of the output. If append_columns is True, the meta should specify just the
            additional columns output by func.
        append_columns : bool
            If True, the output columns should be appended to those in the original catalog.
        infer_nesting : bool
            If True, the function will pack output columns into nested structures based on column names
            adhering to a nested naming scheme. E.g. `nested.b` and `nested.c` will be packed into a
            column called `nested` with columns `b` and `c`. If False, all outputs will be returned as base
            columns.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `Catalog`
            `Catalog` with the results of the function applied to the columns of the frame.

        Notes
        -----
        By default, computing a `reduce` will produce a `NestedFrame` with enumerated
        column names for each returned value of the function. For more useful
        naming, it's recommended to have `func` return a dictionary where each
        key is an output column of the dataframe returned by computing `reduce`.

        Example User Function:

        >>> import numpy as np
        >>> import lsdb
        >>> catalog = lsdb.from_dataframe({"ra":[0, 10], "dec":[5, 15], "mag":[21, 22], "mag_err":[.1, .2]})
        >>> def my_sigma(col1, col2):
        ...    '''reduce will return a NestedFrame with two columns'''
        ...    return {"plus_one": col1+col2, "minus_one": col1-col2}
        >>> meta = {"plus_one": np.float64, "minus_one": np.float64}
        >>> catalog.reduce(my_sigma, 'mag', 'mag_err', meta=meta).compute().reset_index()
                   _healpix_29  plus_one  minus_one
        0  1372475556631677955      21.1       20.9
        1  1389879706834706546      22.2       21.8
        """
        catalog = super().reduce(
            func, *args, meta=meta, append_columns=append_columns, infer_nesting=infer_nesting, **kwargs
        )
        if self.margin is not None:
            catalog.margin = self.margin.reduce(
                func, *args, meta=meta, append_columns=append_columns, infer_nesting=infer_nesting, **kwargs
            )
        return catalog

    def to_hats(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        as_collection: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):
        """Save the catalog to disk in the HATS format. See write_catalog()."""
        self.write_catalog(
            base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            as_collection=as_collection,
            overwrite=overwrite,
            **kwargs,
        )

    def write_catalog(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        as_collection: bool = True,
        overwrite: bool = False,
        **kwargs,
    ):
        """Save the catalog to disk in HATS format.

        Args:
            base_catalog_path (str): Location where catalog is saved to
            catalog_name (str): The name of the catalog to be saved
            default_columns (list[str]): A metadata property with the list of the columns in the
                catalog to be loaded by default. By default, uses the default columns from the
                original hats catalog if they exist.
            as_collection (bool): If True, saves the catalog and its margin as a collection
            overwrite (bool): If True existing catalog is overwritten
            **kwargs: Arguments to pass to the parquet write operations
        """
        if as_collection:
            self._check_unloaded_columns(default_columns)
            io.to_collection(
                self,
                base_collection_path=base_catalog_path,
                catalog_name=catalog_name,
                default_columns=default_columns,
                overwrite=overwrite,
                **kwargs,
            )
        else:
            super().write_catalog(
                base_catalog_path,
                catalog_name=catalog_name,
                default_columns=default_columns,
                overwrite=overwrite,
                create_thumbnail=True,
                **kwargs,
            )
