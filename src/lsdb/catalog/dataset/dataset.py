from typing import List

import dask.dataframe as dd
import hipscat as hc
import pandas as pd
from dask.delayed import Delayed


class Dataset:
    """Base HiPSCat Dataset"""

    def __init__(
        self,
        ddf: dd.core.DataFrame,
        hc_structure: hc.catalog.Dataset,
    ):
        """Initialise a Catalog object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            ddf: Dask DataFrame with the source data of the catalog
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
        """
        self._ddf = ddf
        self.hc_structure = hc_structure

    def __repr__(self):
        return self._ddf.__repr__()

    def _repr_html_(self):
        # pylint: disable=protected-access
        data = self._ddf._repr_data().to_html(max_rows=5, show_dimensions=False, notebook=True)
        return f"<div><strong>lsdb Catalog {self.name}:</strong></div>{data}"

    def compute(self) -> pd.DataFrame:
        """Compute dask distributed dataframe to pandas dataframe"""
        return self._ddf.compute()

    def to_delayed(self, optimize_graph: bool = True) -> List[Delayed]:
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
