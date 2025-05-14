from collections.abc import Sequence

import hats as hc
import nested_pandas as npd
from dask.delayed import Delayed

import lsdb.nested as nd


class Dataset:
    """Base HATS Dataset"""

    def __init__(
        self,
        ddf: nd.NestedFrame,
        hc_structure: hc.catalog.Dataset,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            hc_structure: `hats.Catalog` object with hats metadata of the catalog
        """
        self._ddf = ddf
        self.hc_structure = hc_structure

    def __repr__(self):
        return self._ddf.__repr__()

    def _repr_html_(self):
        data = self._repr_data().to_html(max_rows=5, show_dimensions=False, notebook=True)
        loaded_cols = len(self.columns)
        available_cols = len(self.all_columns)
        return (
            f"<div><strong>lsdb Catalog {self.name}:</strong></div>"
            f"{data}"
            f"<div>{loaded_cols} out of {available_cols} columns in the catalog have been loaded "
            f"<strong>lazily</strong>, meaning no data has been read, only the catalog schema</div>"
        )

    def _repr_data(self):
        # pylint: disable=protected-access
        return self._ddf._repr_data()

    def compute(self) -> npd.NestedFrame:
        """Compute dask distributed dataframe to pandas dataframe"""
        return self._ddf.compute()

    def to_delayed(self, optimize_graph: bool = True) -> list[Delayed]:
        """Get a list of Dask Delayed objects for each partition in the dataset

        Used for more advanced custom operations, but to use again with LSDB, the delayed objects
        must be converted to a Dask DataFrame and used with extra metadata to construct an
        LSDB Dataset.

        Args:
            optimize_graph (bool): If True [default], the graph is optimized before converting into
                ``dask.delayed`` objects.
        """
        return self._ddf.to_delayed(optimize_graph=optimize_graph)

    @property
    def name(self):
        """The name of the catalog"""
        return self.hc_structure.catalog_name

    @property
    def dtypes(self):
        """Returns the datatypes of the columns in the Dataset"""
        return self._ddf.dtypes

    @property
    def columns(self):
        """Returns the columns in the Dataset"""
        return self._ddf.columns

    @property
    def all_columns(self):
        """Returns all columns in the original Dataset"""
        if self.hc_structure.original_schema is None:
            # This case corresponds to Datasets that have not yet been
            # serialized, and thus cannot have a discrepancy between
            # the original schema and the loaded schema.  In this case,
            # this property is equivalent to the columns property.
            return self.columns
        col_names = self.hc_structure.original_schema.names
        if self._ddf.index.name in col_names:
            col_names.remove(self._ddf.index.name)
        return col_names

    @property
    def original_schema(self):
        """Returns the schema of the original Dataset"""
        if self.hc_structure.original_schema is None:
            raise ValueError("Original catalog schema is not available")
        return self.hc_structure.original_schema

    def _check_unloaded_columns(self, column_names: Sequence[str | None] | None):
        """Check the list of given column names for any that are valid
        but unavailable because they were not loaded.
        """
        if not column_names:
            return
        # Quick local optimization
        problematic = set(self.all_columns) - set(self.columns)
        confusing = [name for name in column_names if name and name in problematic]
        if not confusing:
            return
        if len(confusing) == 1:
            confusing_column = confusing[0]
            msg = f"Column `{confusing_column}` is in the catalog but was not loaded."
        else:
            confusing_columns = ", ".join([f"`{c}`" for c in confusing])
            msg = f"Columns {confusing_columns} are in the catalog but were not loaded."
        raise ValueError(msg)
