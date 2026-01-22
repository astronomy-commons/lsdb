# Python 3.9 doesn't support "|" for types
from __future__ import annotations

import dask.dataframe as dd
import nested_pandas as npd
from dask.dataframe.extensions import register_series_accessor
from nested_pandas import NestedDtype


@register_series_accessor("nest")
class DaskNestSeriesAccessor(npd.NestSeriesAccessor):
    """The nested-dask version of the nested-pandas NestSeriesAccessor.

    Note that this has a very limited implementation relative to nested-pandas.

    Parameters
    ----------
    series: dd.series
        A series to tie to the accessor
    """

    def __init__(self, series):
        super().__init__(series)
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        """chcek the validity of the tied series dtype"""
        dtype = series.dtype
        if not isinstance(dtype, NestedDtype):
            raise AttributeError(f"Can only use .nest accessor with a Series of NestedDtype, got {dtype}")

    @property
    def columns(self) -> list[str]:
        """Names of the nested columns"""

        return list(self._series.dtype.column_dtypes)

    def to_lists(self, columns: list[str] | str | None = None) -> dd.DataFrame:
        """Convert nested series into dataframe of list-array columns

        Parameters
        ----------
        columns : list[str] or None, optional
            Names of the columns to include. Default is None, which means all columns.

        Returns
        -------
        dd.DataFrame
            Dataframe of list-arrays.
        """
        return self._series.map_partitions(lambda x: x.nest.to_lists(columns=columns))

    def to_flat(self, columns: list[str] | str | None = None) -> dd.DataFrame:
        """Convert nested series into dataframe of flat arrays

        Parameters
        ----------
        columns : list[str] or None, optional
            Names of the columns to include. Default is None, which means all columns.

        Returns
        -------
        dd.DataFrame
            Dataframe of flat arrays.
        """
        return self._series.map_partitions(lambda x: x.nest.to_flat(columns=columns))

    def clear(self):
        """Clear method implementation"""
        raise NotImplementedError("The 'clear' method is not implemented for DaskNestSeriesAccessor.")
