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
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        """chcek the validity of the tied series dtype"""
        dtype = series.dtype
        if not isinstance(dtype, NestedDtype):
            raise AttributeError(f"Can only use .nest accessor with a Series of NestedDtype, got {dtype}")

    @property
    def fields(self) -> list[str]:
        """Names of the nested columns"""

        return list(self._series.dtype.fields)

    def to_lists(self, fields: list[str] | None = None) -> dd.DataFrame:
        """Convert nested series into dataframe of list-array columns

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        dd.DataFrame
            Dataframe of list-arrays.
        """
        return self._series.map_partitions(lambda x: x.nest.to_lists(fields=fields))

    def to_flat(self, fields: list[str] | None = None) -> dd.DataFrame:
        """Convert nested series into dataframe of flat arrays

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        dd.DataFrame
            Dataframe of flat arrays.
        """
        return self._series.map_partitions(lambda x: x.nest.to_flat(fields=fields))
