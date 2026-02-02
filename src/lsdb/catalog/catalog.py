from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import dask.dataframe as dd
import hats as hc
import nested_pandas as npd
import pandas as pd
from deprecated import deprecated  # type: ignore
from hats.catalog.catalog_collection import CatalogCollection
from hats.catalog.healpix_dataset.healpix_dataset import HealpixDataset as HCHealpixDataset
from hats.catalog.index.index_catalog import IndexCatalog as HCIndexCatalog
from hats.pixel_math import HealpixPixel
from pandas._typing import Renamer
from typing_extensions import Self
from upath import UPath

from lsdb import io
from lsdb.catalog.association_catalog import AssociationCatalog
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.catalog.map_catalog import MapCatalog
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.core.search.index_search import IndexSearch
from lsdb.dask.concat_catalog_data import _assert_same_ra_dec, concat_catalog_data, handle_margins_for_concat
from lsdb.dask.crossmatch_catalog_data import crossmatch_catalog_data, crossmatch_catalog_data_nested
from lsdb.dask.join_catalog_data import (
    join_catalog_data_nested,
    join_catalog_data_on,
    join_catalog_data_through,
    merge_asof_catalog_data,
)
from lsdb.dask.merge_catalog_functions import DEFAULT_SUFFIX_METHOD, create_merged_catalog_info
from lsdb.dask.merge_map_catalog_data import merge_map_catalog_data
from lsdb.io.schema import get_arrow_schema
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.nested.core import NestedFrame
from lsdb.types import DaskDFPixelMap


def _default_suffixes(left_name: str, right_name: str) -> tuple[str, str]:
    """Return the default pair of suffixes for left/right catalog names."""
    return (f"_{left_name}", f"_{right_name}")


# pylint: disable=protected-access,too-many-public-methods, too-many-lines
class Catalog(HealpixDataset):
    """LSDB Catalog to perform analysis of sky catalogs and efficient spatial operations."""

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
        margin: MarginCatalog or None, default None
            The margin catalog.
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
        Catalog
            A catalog that contains the data from the original catalog that complies with the query
            expression. If a margin exists, it is filtered according to the same query expression.

        Examples
        --------
        Filter a small synthetic catalog using a pandas-style query string:

        >>> import lsdb
        >>> from lsdb.nested.datasets import generate_data
        >>> nf = generate_data(1000, 5, seed=0, ra_range=(0.0, 300.0), dec_range=(-50.0, 50.0))
        >>> catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]])
        >>> filtered = catalog.query("ra < 100 and dec > 0")
        >>> filtered.compute().head()
        """
        catalog = super().query(expr)
        if self.margin is not None:
            catalog.margin = self.margin.query(expr)
        return catalog

    def rename(self, columns: Renamer) -> Catalog:
        """Renames catalog columns (not indices) and that of its margin if it exists using a
        dictionary or function mapping.

        Parameters
        ----------
        columns : dict-like or function
            Transformations to apply to column names.

        Returns
        -------
        Catalog
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
        *,
        n_neighbors: int | None = None,
        radius_arcsec: float | None = None,
        min_radius_arcsec: float | None = None,
        algorithm: AbstractCrossmatchAlgorithm | None = None,
        output_catalog_name: str | None = None,
        require_right_margin: bool = False,
        how: str = "inner",
        suffixes: tuple[str, str] | None = None,
        suffix_method: str | None = None,
        log_changes: bool = True,
    ) -> Catalog:
        # pylint:disable=unused-argument
        """Perform a cross-match between two catalogs

        The pixels from each catalog are aligned via a `PixelAlignment`, and cross-matching is
        performed on each pair of overlapping pixels. The resulting catalog will have partitions
        matching an inner pixel alignment - using pixels that have overlap in both input catalogs
        and taking the smallest of any overlapping pixels.

        The resulting catalog will be partitioned using the left catalog's ra and dec, and the
        index for each row will be the same as the index from the corresponding row in the left
        catalog's index.

        Parameters
        ----------
        other : Catalog
            The right catalog to cross-match against
        n_neighbors : int, default 1
            The number of neighbors to find within each point.
        radius_arcsec : float, default 1.0
            The threshold distance in arcseconds beyond which neighbors are not added.
        min_radius_arcsec : float, default 0.0
            The threshold distance in arcseconds beyond which neighbors are added.
        algorithm : AbstractCrossmatchAlgorithm | None, default `KDTreeCrossmatch`
            The instance of an algorithm used to perform the crossmatch. If None,
            the default KDTree crossmatch algorithm is used. If specified, the
            algorithm is defined by subclassing `AbstractCrossmatchAlgorithm`.

            Default algorithm:
                - `KdTreeCrossmatch`: find the k-nearest neighbors using a kd_tree

            Custom algorithm:
                To specify a custom algorithm, write a class that subclasses the
                `AbstractCrossmatchAlgorithm` class, and either overwrite the `crossmatch`
                or the `perform_crossmatch` function.

                The function should be able to perform a crossmatch on two pandas DataFrames
                from a partition from each catalog. It should return two 1d numpy arrays of equal lengths
                with the indices of the matching rows from the left and right dataframes, and a dataframe
                with any extra columns generated by the crossmatch algorithm, also with the same length.
                These columns are specified in {AbstractCrossmatchAlgorithm.extra_columns}, with
                their respective data types, by means of an empty pandas dataframe. As an example,
                the KdTreeCrossmatch algorithm outputs a "_dist_arcsec" column with the distance between
                data points. Its extra_columns attribute is specified as follows::

                    pd.DataFrame({"_dist_arcsec": pd.Series(dtype=np.dtype("float64"))})

                The `crossmatch`/`perform_crossmatch` methods will receive an instance of `CrossmatchArgs`
                which includes the partitions and respective pixel information::

                    - left_df: npd.NestedFrame
                    - right_df: npd.NestedFrame
                    - left_order: int
                    - left_pixel: int
                    - right_order: int
                    - right_pixel: int
                    - left_catalog_info: hc.catalog.TableProperties
                    - right_catalog_info: hc.catalog.TableProperties
                    - right_margin_catalog_info: hc.catalog.TableProperties

                Include any algorithm-specific parameters in the initialization of your object.
                These parameters should be validated in `AbstractCrossmatchAlgorithm.validate`,
                by overwriting the method.

        output_catalog_name : str, default {left_name}_x_{right_name}
            The name of the resulting catalog.
        require_right_margin : bool, default False
            If true, raises an error if the right margin is missing which could
            lead to incomplete crossmatches.
        how : str
            How to handle the crossmatch of the two catalogs.
            One of {'left', 'inner'}; defaults to 'inner'.
        suffixes : Tuple[str,str] or None
            A pair of suffixes to be appended to the end of each column
            name when they are joined. Default uses the name of the catalog for the suffix.
        suffix_method : str or None, default "all_columns"
            Method to use to add suffixes to columns. Options are:

            - "overlapping_columns": only add suffixes to columns that are present in both catalogs
            - "all_columns": add suffixes to all columns from both catalogs

            .. warning:: This default will change to "overlapping_columns" in a future release.

        log_changes : bool, default True
            If True, logs an info message for each column that is being renamed.
            This only applies when suffix_method is 'overlapping_columns'.

        Returns
        -------
        Catalog
            A Catalog with the data from the left and right catalogs merged with one row for each
            pair of neighbors found from cross-matching.
            The resulting table contains all columns from the left and right catalogs with their
            respective suffixes and, whenever specified, a set of extra columns generated by the
            crossmatch algorithm.

        Raises
        ------
        TypeError
            If the `other` catalog is not of type `Catalog`
        ValueError
            If both the kwargs for the default algorithm and an `algorithm` are specified.
            If the `suffixes` provided is not a tuple of two strings.
            If the right catalog has no margin and `require_right_margin` is True.
        """
        if not isinstance(other, Catalog):
            raise TypeError(
                f"Expected `other` to be a Catalog instance, got {type(other)}. "
                "You may want `lsdb.crossmatch(frame_or_catalog, frame_or_catalog)` instead."
            )

        default_kwargs = {
            k: v
            for k, v in locals().items()
            if k in ("radius_arcsec", "n_neighbors", "min_radius_arcsec") and v is not None
        }
        if not algorithm:
            algorithm = KdTreeCrossmatch(**default_kwargs)
        elif any(default_kwargs.values()):
            raise ValueError(f"If you specify `algorithm`, do not set {list(default_kwargs.keys())}")

        if suffixes is None:
            suffixes = _default_suffixes(self.name, other.name)
        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")
        if suffix_method is None:
            suffix_method = DEFAULT_SUFFIX_METHOD
            warnings.warn(
                "The default suffix behavior will change from applying suffixes to all columns to only "
                "applying suffixes to overlapping columns in a future release."
                "To maintain the current behavior, explicitly set `suffix_method='all_columns'`. "
                "To change to the new behavior, set `suffix_method='overlapping_columns'`.",
                FutureWarning,
            )
        if other.margin is None and require_right_margin:
            raise ValueError("Right catalog margin cache is required for cross-match.")
        if output_catalog_name is None:
            output_catalog_name = f"{self.name}_x_{other.name}"

        ddf, ddf_map, alignment = crossmatch_catalog_data(
            self,
            other,
            algorithm,
            how,
            suffixes,
            suffix_method,
            log_changes,
        )
        new_catalog_info = create_merged_catalog_info(
            self,
            other,
            output_catalog_name,
            suffixes,
            suffix_method,
        )
        hc_catalog = self.hc_structure.__class__(
            new_catalog_info, alignment.pixel_tree, schema=get_arrow_schema(ddf), moc=alignment.moc
        )
        return self.__class__(ddf, ddf_map, hc_catalog)

    def crossmatch_nested(
        self,
        other: Catalog,
        *,
        n_neighbors: int | None = None,
        radius_arcsec: float | None = None,
        min_radius_arcsec: float | None = None,
        algorithm: AbstractCrossmatchAlgorithm | None = None,
        output_catalog_name: str | None = None,
        require_right_margin: bool = False,
        nested_column_name: str | None = None,
    ) -> Catalog:
        # pylint:disable=unused-argument
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

        Parameters
        ----------
        other : Catalog
            The right catalog to cross-match against
        n_neighbors : int, default 1
            The number of neighbors to find within each point.
        radius_arcsec : float, default 1.0
            The threshold distance in arcseconds beyond which neighbors are not added.
        min_radius_arcsec : float, default 0.0
            The threshold distance in arcseconds beyond which neighbors are added.
        algorithm : AbstractCrossmatchAlgorithm | None, default `KDTreeCrossmatch`
            The instance of an algorithm used to perform the crossmatch. If None,
            the default KDTree crossmatch algorithm is used. If specified, the
            algorithm is defined by subclassing `AbstractCrossmatchAlgorithm`.

            Default algorithm:
                - `KdTreeCrossmatch`: find the k-nearest neighbors using a kd_tree

            Custom algorithm:
                To specify a custom algorithm, write a class that subclasses the
                `AbstractCrossmatchAlgorithm` class, and either overwrite the `crossmatch`
                or the `perform_crossmatch` function.

                The function should be able to perform a crossmatch on two pandas DataFrames
                from a partition from each catalog. It should return two 1d numpy arrays of equal lengths
                with the indices of the matching rows from the left and right dataframes, and a dataframe
                with any extra columns generated by the crossmatch algorithm, also with the same length.
                These columns are specified in {AbstractCrossmatchAlgorithm.extra_columns}, with
                their respective data types, by means of an empty pandas dataframe. As an example,
                the KdTreeCrossmatch algorithm outputs a "_dist_arcsec" column with the distance between
                data points. Its extra_columns attribute is specified as follows::

                    pd.DataFrame({"_dist_arcsec": pd.Series(dtype=np.dtype("float64"))})

                The `crossmatch`/`perform_crossmatch` methods will receive an instance of `CrossmatchArgs`
                which includes the partitions and respective pixel information::

                    - left_df: npd.NestedFrame
                    - right_df: npd.NestedFrame
                    - left_order: int
                    - left_pixel: int
                    - right_order: int
                    - right_pixel: int
                    - left_catalog_info: hc.catalog.TableProperties
                    - right_catalog_info: hc.catalog.TableProperties
                    - right_margin_catalog_info: hc.catalog.TableProperties

                Include any algorithm-specific parameters in the initialization of your object.
                These parameters should be validated in `AbstractCrossmatchAlgorithm.validate`,
                by overwriting the method.

        output_catalog_name : str, default {left_name}_x_{right_name}
            The name of the resulting catalog.
        require_right_margin : bool, default False
            If true, raises an error if the right margin is missing which could
            lead to incomplete crossmatches.
        nested_column_name : str, default uses the name of the right catalog
            The name of the nested column that will contain the crossmatched rows
            from the right catalog.

        Returns
        -------
        Catalog
            A Catalog with the data from the left and right catalogs joined with the cross-matched rows from
            the right catalog added in a new nested column.
            The resulting table contains all columns from the left catalog and a new nested column with all
            the columns from the right catalog and any extra columns generated by the crossmatch algorithm.

        Raises
        ------
        ValueError
            If both the kwargs for the default algorithm and an `algorithm` are specified.
            If the right catalog has no margin and  `require_right_margin` is True.
        """
        default_kwargs = {
            k: v
            for k, v in locals().items()
            if k in ("radius_arcsec", "n_neighbors", "min_radius_arcsec") and v is not None
        }
        if not algorithm:
            algorithm = KdTreeCrossmatch(**default_kwargs)
        elif any(default_kwargs.values()):
            raise ValueError(f"If you specify `algorithm`, do not set {list(default_kwargs.keys())}")

        if nested_column_name is None:
            nested_column_name = other.name
        if other.margin is None and require_right_margin:
            raise ValueError("Right catalog margin cache is required for cross-match.")
        if output_catalog_name is None:
            output_catalog_name = f"{self.name}_x_{other.name}"

        ddf, ddf_map, alignment = crossmatch_catalog_data_nested(self, other, algorithm, nested_column_name)
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
        *,
        ignore_empty_margins: bool = False,
        **kwargs,
    ) -> Catalog:
        """Concatenate two catalogs by aligned HEALPix pixels.

        Parameters
        ----------
        other : Catalog
            Catalog to concatenate with.
        ignore_empty_margins : bool, default False
            If True, keep the available margin when only one side has it
            (treated as incomplete). If False, drop margins when only one
            side has them. Defaults to False.
        **kwargs
            Extra arguments forwarded to internal `pandas.concat`.

        Returns
        -------
        Catalog
            New catalog with OUTER pixel alignment. If both inputs have a
            margin — or if `ignore_empty_margins=True` and at least one side has it —
            the result includes a concatenated margin dataset.

        Raises
        ------
        ValueError
            If RA/Dec column names differ between the input catalogs, or
            between a catalog and its own margin.
        """
        # Fail fast if RA/Dec columns differ between the two catalogs.
        _assert_same_ra_dec(self, other, context="Catalog concat")

        # Delegate margin handling to helper (which also validates catalog vs margin)
        margin = handle_margins_for_concat(
            self,
            other,
            ignore_empty_margins=ignore_empty_margins,
            **kwargs,
        )

        # Main catalog concatenation
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

        Parameters
        ----------
        map_catalog : MapCatalog
            The continuous map to merge.
        func : Callable
            The function applied to each catalog partition, which will be called with:
            `func(catalog_partition: npd.NestedFrame, map_partition: npd.NestedFrame, `
            ` healpix_pixel: HealpixPixel, *args, **kwargs)`
            with the additional args and kwargs passed to the `merge_map` function.
        *args :
            Additional positional arguments to call `func` with.
        meta : pd.DataFrame | pd.Series | Dict | Iterable | Tuple | None, default None
            An empty pandas DataFrame that
            has columns matching the output of the function applied to the catalog partition. Other types
            are accepted to describe the output dataframe format, for full details see the dask
            documentation https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
            If meta is None (default), LSDB will try to work out the output schema of the function by
            calling the function with an empty DataFrame. If the function does not work with an empty
            DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
            will generate empty partitions, though these can be removed by calling the
            `Catalog.prune_empty_partitions` method.
        **kwargs
            Additional keyword args to pass to the function. These are passed to the Dask DataFrame
            `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
            `transform_divisions` will be passed through and work as described in the dask documentation
            https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html

        Returns
        -------
        Catalog
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

        Parameters
        ----------
        values : dict[str, Any]
            The mapping of field names (as string) to their search values.
            Values can be single values or lists of values (but only one list of values is
            allowed per search). If a list is specified, the search will return rows that match
            any of the values in the list.
        index_catalogs : dict[str, str | HCIndexCatalog] | None, default None
            The mapping of field names (as string)
            to their respective index catalog paths or instance of `HCIndexCatalog`. Use this
            argument to specify index catalogs for stand-alone catalogs or for collections where
            there is no index catalog for the fields you are querying for.
        fine : bool, default True
            If True, the rows of the partitions where a column match occurred are
            filtered. If False, all the rows of those partitions are kept. Defaults to True.

        Returns
        -------
        Catalog
            A new Catalog containing the results of the column match.

        Raises
        ------
        ValueError
            If no values were provided for the search.
            If more than one column contains a list of values to search for.
        TypeError
            If index catalog for field is not of type `HCIndexCatalog`.

        Examples
        --------

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

        Parameters
        ----------
        search : AbstractSearch
            Instance of AbstractSearch.

        Returns
        -------
        Catalog
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
        compute_single_partition: bool = False,
        partition_index: int | HealpixPixel | None = None,
        **kwargs,
    ) -> Catalog | dd.Series:
        """Applies a function to each partition in the catalog and respective margin.

        The ra and dec of each row is assumed to remain unchanged. If the function returns a DataFrame,
        an LSDB Catalog is constructed and its respective margin is updated accordingly, if it exists.
        Otherwise, only the main catalog Dask object is returned.

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
            An empty pandas DataFrame that
            has columns matching the output of the function applied to a partition. Other types are
            accepted to describe the output dataframe format, for full details see the dask documentation
            https://blog.dask.org/2022/08/09/understanding-meta-keyword-argument
            If meta is None (default), LSDB will try to work out the output schema of the function by
            calling the function with an empty DataFrame. If the function does not work with an empty
            DataFrame, this will raise an error and meta must be set. Note that some operations in LSDB
            will generate empty partitions, though these can be removed by calling the
            `Catalog.prune_empty_partitions` method.
        include_pixel : bool, default False
            Whether to pass the Healpix Pixel of the partition as a `HealpixPixel`
            object to the second positional argument of the function
        compute_single_partition : bool, default False
            If true, runs the function on a single partition only in the local thread, without going through
            dask. This is useful for testing and debugging functions on a single partition, as all normal
            debugging tools can be used. Note that when this is true, which partition is computed is
            determined by the `partition_index` parameter.
        partition_index : int | HealpixPixel | None, default None
            The index of the partition to compute when compute_single_partition is True. Also accepts a
            HealpixPixel object to specify the partition by its HEALPix order and pixel.
            If None, defaults to 0.
        **kwargs
            Additional keyword args to pass to the function. These are passed to the Dask DataFrame
            `dask.dataframe.map_partitions` function, so any of the dask function's keyword args such as
            `transform_divisions` will be passed through and work as described in the dask documentation
            https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html

        Returns
        -------
        Catalog | dd.Series
            A new catalog with each partition replaced with the output of the function applied to the original
            partition. If the function returns a non dataframe output, a dask Series will be returned.

        Examples
        --------
        Apply a function to each partition (e.g., add a derived column):

        >>> import lsdb
        >>> from lsdb.nested.datasets import generate_data
        >>> nf = generate_data(1000, 5, seed=0, ra_range=(0.0, 300.0), dec_range=(-50.0, 50.0))
        >>> catalog = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]])
        >>> def add_flag(df):
        ...     return df.assign(in_north=df["dec"] > 0)
        >>> catalog2 = catalog.map_partitions(add_flag)
        >>> catalog2.compute().head()
        """
        catalog = super().map_partitions(
            func,
            *args,
            meta=meta,
            include_pixel=include_pixel,
            compute_single_partition=compute_single_partition,
            partition_index=partition_index,
            **kwargs,
        )
        if isinstance(catalog, Catalog) and self.margin is not None:
            # For single partition updates, we need to update the margin for that partition only
            if compute_single_partition:
                # Get the corresponding pixel for this partition
                pixel = catalog.get_healpix_pixels()[0]

                # Update the margin for this pixel only
                if pixel in self.margin._ddf_pixel_map:
                    margin_partition_index = self.margin.get_partition_index(pixel.order, pixel.pixel)
                    catalog.margin = self.margin.map_partitions(
                        func,
                        *args,
                        meta=meta,
                        include_pixel=include_pixel,
                        compute_single_partition=True,
                        partition_index=margin_partition_index,
                        **kwargs,
                    )  # type: ignore[assignment]
            else:
                # Update all margins as before
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

        Parameters
        ----------
        other : Catalog
            The right catalog to merge with.
        how : str
            How to handle the merge of the two catalogs.
            One of {'left', 'right', 'outer', 'inner', 'leftsemi'}, default 'inner'
        on : str | List
            Column or index names to join on. Defaults to the
            intersection of columns in both Dataframes if on is None and not
            merging on indexes.
        left_on : str | List
            Column to join on the left Dataframe. Lists are
            supported if their length is one.
        right_on : str | List
            Column to join on the right Dataframe. Lists are
            supported if their length is one.
        left_index : bool, default False
            Use the index of the left Dataframe as the join key.
        right_index : bool, default False
            Use the index of the right Dataframe as the join key.
        suffixes : tuple[str,str]
            A pair of suffixes to be appended to the end of each column name
            when they are joined. Defaults to using the name of the catalog
            for the suffix.

        Returns
        -------
        Catalog
            A new Dask Dataframe containing the data points that result from the merge
            of the two catalogs.
        """
        if suffixes is None:
            suffixes = _default_suffixes(self.name, other.name)
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
        suffix_method: str | None = None,
        log_changes: bool = True,
    ):
        """Uses the pandas `merge_asof` function to merge two catalogs on their indices by distance of keys

        Must be along catalog indices, and does not include margin caches, meaning results may be incomplete
        for merging points.

        This function is intended for use in special cases such as Dust Map Catalogs, for general merges,
        the ``crossmatch`` and ``join`` functions should be used.

        Parameters
        ----------
        other : lsdb.Catalog
            The right catalog to merge to
        suffixes : tuple[str,str]
            The suffixes to apply to each partition's column names
        direction : str, default "backward"
            The direction to perform the merge_asof
        output_catalog_name : str
            The name of the resulting catalog to be stored in metadata
        suffix_method : str, default "all_columns"
            Method to use to add suffixes to columns. Options are:

            - "overlapping_columns": only add suffixes to columns that are present in both catalogs
            - "all_columns": add suffixes to all columns from both catalogs

            .. warning:: This default will change to "overlapping_columns" in a future release.

        log_changes : bool, default True
            If True, logs an info message for each column that is being renamed.
            This only applies when suffix_method is 'overlapping_columns'.

        Returns
        -------
        Catalog
            A new catalog with the columns from each of the input catalogs with their respective suffixes
            added, and the rows merged using merge_asof on the specified columns.
        """
        if suffixes is None:
            suffixes = _default_suffixes(self.name, other.name)

        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")

        if suffix_method is None:
            suffix_method = DEFAULT_SUFFIX_METHOD
            warnings.warn(
                "The default suffix behavior will change from applying suffixes to all columns to only "
                "applying suffixes to overlapping columns in a future release."
                "To maintain the current behavior, explicitly set `suffix_method='all_columns'`. "
                "To change to the new behavior, set `suffix_method='overlapping_columns'`.",
                FutureWarning,
            )

        ddf, ddf_map, alignment = merge_asof_catalog_data(
            self,
            other,
            suffixes=suffixes,
            direction=direction,
            suffix_method=suffix_method,
            log_changes=log_changes,
        )

        if output_catalog_name is None:
            output_catalog_name = (
                f"{self.hc_structure.catalog_info.catalog_name}_merge_asof_"
                f"{other.hc_structure.catalog_info.catalog_name}"
            )

        new_catalog_info = create_merged_catalog_info(
            self,
            other,
            output_catalog_name,
            suffixes,
            suffix_method,
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
        suffix_method: str | None = None,
        log_changes: bool = True,
    ) -> Catalog:
        """Perform a spatial join to another catalog

        Joins two catalogs together on a shared column value, merging rows where they match.

        This is an inner join: only rows with matching join keys are returned (unmatched rows are dropped).

        The operation only joins data from matching partitions, and does not join rows that have a matching
        column value but are in separate partitions in the sky. For a more general join, see the `merge`
        function.

        Parameters
        ----------
        other : Catalog
            The right catalog to join to
        left_on : str
            The name of the column in the left catalog to join on
        right_on : str
            The name of the column in the right catalog to join on
        through : AssociationCatalog
            An association catalog that provides the alignment
            between pixels and individual rows.
        suffixes : tuple[str,str]
            Suffixes to apply to the columns of each table
        output_catalog_name : str
            The name of the resulting catalog to be stored in metadata
        suffix_method : str, default "all_columns"
            Method to use to add suffixes to columns. Options are:

            - "overlapping_columns": only add suffixes to columns that are present in both catalogs
            - "all_columns": add suffixes to all columns from both catalogs

            .. warning:: This default will change to "overlapping_columns" in a future release.

        log_changes : bool, default True
            If True, logs an info message for each column that is being renamed.
            This only applies when suffix_method is 'overlapping_columns'.

        Returns
        -------
        Catalog
            A new catalog with the columns from each of the input catalogs with their respective suffixes
            added, and the rows merged on the specified columns.

        Examples
        --------
        Join two catalogs on a shared key within the same sky partitions:

        >>> import lsdb
        >>> from lsdb.nested.datasets import generate_data
        >>> nf = generate_data(1000, 5, seed=0, ra_range=(0.0, 300.0), dec_range=(-50.0, 50.0))
        >>> base = lsdb.from_dataframe(nf.compute()[["ra", "dec", "id"]])
        >>> left = base.rename({"ra": "ra_left", "dec": "dec_left"})
        >>> right = base.rename({"ra": "ra_right", "dec": "dec_right", "id": "id_right"}).map_partitions(
        ...     lambda df: df.assign(right_flag=True)
        ... )
        >>> joined = left.join(right, left_on="id", right_on="id_right", suffix_method="overlapping_columns")
        >>> joined.compute().head()
        """
        if suffixes is None:
            suffixes = _default_suffixes(self.name, other.name)

        if len(suffixes) != 2:
            raise ValueError("`suffixes` must be a tuple with two strings")

        if suffix_method is None:
            suffix_method = DEFAULT_SUFFIX_METHOD
            warnings.warn(
                "The default suffix behavior will change from applying suffixes to all columns to only "
                "applying suffixes to overlapping columns in a future release."
                "To maintain the current behavior, explicitly set `suffix_method='all_columns'`. "
                "To change to the new behavior, set `suffix_method='overlapping_columns'`.",
                FutureWarning,
            )

        self._check_unloaded_columns([left_on, right_on])

        if through is not None:
            ddf, ddf_map, alignment = join_catalog_data_through(
                self, other, through, suffixes, suffix_method=suffix_method, log_changes=log_changes
            )
        else:
            if left_on is None or right_on is None:
                raise ValueError("Either both of left_on and right_on, or through must be set")
            if left_on not in self._ddf.columns:
                raise ValueError("left_on must be a column in the left catalog")
            if right_on not in other._ddf.columns:
                raise ValueError("right_on must be a column in the right catalog")
            ddf, ddf_map, alignment = join_catalog_data_on(
                self, other, left_on, right_on, suffixes, suffix_method=suffix_method, log_changes=log_changes
            )

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = create_merged_catalog_info(
            self,
            other,
            output_catalog_name,
            suffixes,
            suffix_method,
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
        how: str = "inner",
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

        Parameters
        ----------
        other : Catalog
            The right catalog to join to
        left_on : str
            The name of the column in the left catalog to join on
        right_on : str
            The name of the column in the right catalog to join on
        nested_column_name : str
            The name of the nested column in the resulting dataframe storing the
            joined columns in the right catalog. (Default: name of right catalog)
        output_catalog_name : str
            The name of the resulting catalog to be stored in metadata
        how : str, {'inner', 'left'}, default 'inner'
            How to handle the alignment

        Returns
        -------
        Catalog
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
            self, other, left_on, right_on, nested_column_name=nested_column_name, how=how
        )

        if output_catalog_name is None:
            output_catalog_name = self.hc_structure.catalog_info.catalog_name

        new_catalog_info = self.hc_structure.catalog_info.copy_and_update(
            catalog_name=output_catalog_name, total_rows=None
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

        Parameters
        ----------
        base_columns : list-like or None
            Any columns that have non-list values in the input catalog.
            These will simply be kept as identical columns in the result. If None, is inferred to be
            all columns in the input catalog that are not considered list-value columns.
        list_columns : list-like or None
            The list-value columns that should be packed into a nested
            column. All columns in the list will attempt to be packed into a single nested column with
            the name provided in ``nested_name``. All columns in list_columns must have pyarrow list
            dtypes, otherwise the operation will fail. If None, is defined as all columns not in
            ``base_columns``.

        Returns
        -------
        Catalog
            A new catalog with specified list columns nested into a new nested column.

        Notes
        -----
        As noted above, all columns in `list_columns` must have a pyarrow
        ListType dtype. This is needed for proper meta propagation. To convert
        a list column to this dtype, you can use this command structure::

            nf= nf.astype({"colname": pd.ArrowDtype(pa.list_(pa.int64()))})

        Where pa.int64 above should be replaced with the correct dtype of the
        underlying data accordingly.
        Additionally, it's a known issue in Dask
        (https://github.com/dask/dask/issues/10139) that columns with list
        values will by default be converted to the string type. This will
        interfere with the ability to recast these to pyarrow lists. We
        recommend setting the following dask config setting to prevent this::

            dask.config.set({"dataframe.convert-string":False})
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

    def map_rows(
        self,
        func,
        columns=None,
        *,
        row_container="dict",
        output_names=None,
        infer_nesting=True,
        append_columns=False,
        meta=None,
        **kwargs,
    ) -> Catalog:
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
            positional arguments corresponding to the columns specified in `args`. (Default value = "dict")
        output_names : None | str | list of str
            Specifies the names of the output columns in the resulting NestedFrame. If None, the function
            will return whatever names the user function returns. If specified will override any names
            returned by the user function provided the number of names matches the number of outputs. When not
            specified and the user function returns values without names (e.g. a list or tuple), the output
            columns will be enumerated (e.g. "0", "1", ...). (Default value = None)
        infer_nesting : bool, default True
            If True, the function will pack output columns into nested
            structures based on column names adhering to a nested naming
            scheme. E.g. "nested.b" and "nested.c" will be packed into a column
            called "nested" with columns "b" and "c". If False, all outputs
            will be returned as base columns. Note that this will trigger off of names specified in
            `output_names` in addition to names returned by the user function. (Default value = True)
        append_columns : bool, default False
            if True, the output columns should be appended to those in the original NestedFrame
        meta : dataframe or series-like, optional, default None
            The dask meta of the output. If append_columns is True, the meta should specify just the
            additional columns output by func.
        kwargs : keyword arguments, optional
            Keyword arguments to pass to the function.

        Returns
        -------
        `Catalog`
            `Catalog` with the results of the function applied to the columns of the frame.

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

        >>> import numpy as np
        >>> import lsdb
        >>> import pandas as pd
        >>> catalog = lsdb.from_dataframe(pd.DataFrame({"ra":[0, 10], "dec":[5, 15],
        ...                                             "mag":[21, 22], "mag_err":[.1, .2]}))

        Writing a function that takes a row as a dictionary:

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
        catalog = super().map_rows(
            func,
            columns=columns,
            row_container=row_container,
            output_names=output_names,
            infer_nesting=infer_nesting,
            append_columns=append_columns,
            meta=meta,
            **kwargs,
        )
        if self.margin is not None:
            catalog.margin = self.margin.map_rows(
                func,
                columns=columns,
                row_container=row_container,
                output_names=output_names,
                infer_nesting=infer_nesting,
                append_columns=append_columns,
                meta=meta,
                **kwargs,
            )
        return catalog

    @deprecated(
        version="0.7.3", reason="`to_hats` will be removed in the future, " "use `write_catalog` instead."
    )
    def to_hats(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        as_collection: bool = True,
        overwrite: bool = False,
        error_if_empty: bool = True,
        **kwargs,
    ):
        """Save the catalog to disk in the HATS format. See write_catalog()."""
        self.write_catalog(
            base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            as_collection=as_collection,
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
        as_collection: bool = True,
        overwrite: bool = False,
        create_thumbnail: bool = True,
        error_if_empty: bool = True,
        **kwargs,
    ):
        """Save the catalog to disk in HATS format.

        Parameters
        ----------
        base_catalog_path : str | Path | UPath,
            Location where catalog is saved to
        catalog_name : str
            The name of the catalog to be saved
        default_columns : list[str]
            A metadata property with the list of the columns in the
            catalog to be loaded by default. By default, uses the default columns from the
            original hats catalog if they exist.
        as_collection : bool, default True
            If True, saves the catalog and its margin as a collection
        overwrite : bool, default False
            If True existing catalog is overwritten
        error_if_empty : bool, default True
            If True, raises an error if the catalog is empty.
        **kwargs
            Arguments to pass to the parquet write operations
        """
        if as_collection:
            self._check_unloaded_columns(default_columns)
            io.to_collection(
                self,
                base_collection_path=base_catalog_path,
                catalog_name=catalog_name,
                default_columns=default_columns,
                overwrite=overwrite,
                error_if_empty=error_if_empty,
                create_thumbnail=create_thumbnail,
                **kwargs,
            )
        else:
            super().write_catalog(
                base_catalog_path,
                catalog_name=catalog_name,
                default_columns=default_columns,
                overwrite=overwrite,
                create_thumbnail=create_thumbnail,
                error_if_empty=error_if_empty,
                **kwargs,
            )
