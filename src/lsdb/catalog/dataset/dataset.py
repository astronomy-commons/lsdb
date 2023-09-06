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
            ddf_pixel_map: Dictionary mapping HEALPix order and pixel to partition index of ddf
            hc_structure: `hipscat.Catalog` object with hipscat metadata of the catalog
        """
        self._ddf = ddf
        self.hc_structure = hc_structure

    def __repr__(self):
        return self._ddf.__repr__()

    def _repr_html_(self):
        return self._ddf._repr_html_()  # pylint: disable=protected-access

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
    def dtypes(self):
        """Returns the datatypes of the columns in the Dataset"""
        return self._ddf.dtypes
