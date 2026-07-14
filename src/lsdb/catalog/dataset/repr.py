from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from dask.dataframe.core import _repr_data_series
from hats.pixel_math import HealpixPixel
from human_readable import file_size, int_comma

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


_MAX_STR_WIDTH = 10
_REPR_MAX_ROWS = 5


class CatalogRepr:
    """Renders the text and HTML representations of a Catalog"""

    def __init__(self, catalog: HealpixDataset):
        self.catalog = catalog

    def text(self) -> str:
        """Returns the text representation of a catalog"""
        return self._render_text(*self._data())

    def html(self) -> str:
        """Returns the HTML representation of a catalog"""
        return self._render_html(*self._data())

    def mimebundle(self) -> dict[str, str]:
        """Return both the text and HTML representations for notebook display.

        IPython calls this in preference to ``__repr__``/``_repr_html_`` and
        picks the richest format the frontend supports (``text/html`` in a
        notebook, ``text/plain`` in a terminal).

        If this method is not defined, both ``__repr__`` and ``_repr_html_``
        are invoked, leading to double-rendering. See:
        - https://ipython.readthedocs.io/en/stable/config/integrating.html
        - https://github.com/ipython/ipython/issues/9771
        - https://github.com/ibis-project/ibis/pull/10002
        """
        data, has_statistics = self._data()
        return {
            "text/plain": self._render_text(data, has_statistics),
            "text/html": self._render_html(data, has_statistics),
        }

    def _render_text(self, data: pd.DataFrame, has_statistics: bool) -> str:
        body = data.to_string(max_rows=_REPR_MAX_ROWS, show_dimensions=False)
        footer = "\n".join(self._footer_lines(has_statistics))
        return f"lsdb Catalog {self.catalog.name}:\n{body}\n{footer}"

    def _render_html(self, data: pd.DataFrame, has_statistics: bool) -> str:
        body = data.to_html(max_rows=_REPR_MAX_ROWS, show_dimensions=False, notebook=True)
        footer = "".join(f"<div>{line}</div>" for line in self._footer_lines(has_statistics))
        return f"<div><strong>lsdb Catalog {self.catalog.name}:</strong></div>{body}{footer}"

    def _footer_lines(self, has_statistics: bool) -> list[str]:
        loaded_cols = len(self.catalog.columns)
        available_cols = len(self.catalog.all_columns)
        lines = [
            f"{loaded_cols} out of {available_cols} available columns in the catalog have been "
            f"loaded lazily, meaning no data has been read, only the catalog schema"
        ]
        est_size = self.catalog.est_size()
        if est_size is not None:
            lines.append(f"This catalog has an estimated size of {file_size(int(est_size * 1024))}")
        if has_statistics:
            lines.append("Statistics for each column are defined by {min}..{max} ranges")
        return lines

    def _data(self) -> tuple[pd.DataFrame, bool]:
        """Build the repr DataFrame and report whether it renders per-column statistics."""
        meta = self.catalog.meta
        index = self._divisions
        cols = meta.columns
        if len(cols) == 0:
            return pd.DataFrame([[]] * len(index), columns=cols, index=index), False
        pixel_stats = self._pixel_stats(cols, index) if self._show_statistics else None
        if pixel_stats is None:
            placeholder = pd.concat([_repr_data_series(s, index=index) for _, s in meta.items()], axis=1)
            return placeholder, False
        return _repr_data_min_max(meta, index, pixel_stats), True

    def _pixel_stats(self, cols, index) -> pd.DataFrame | None:
        """Fetch per-pixel min/max stats for the pixels visible in the repr"""
        try:
            non_nested_cols = [c for c in cols if c not in self.catalog.nested_columns]
            if len(non_nested_cols) == 0:
                return None
            pixels = [p for p in index if isinstance(p, HealpixPixel)]
            return self.catalog.per_partition_statistics(
                use_default_columns=False,
                exclude_hats_columns=True,
                include_columns=non_nested_cols,
                include_stats=["min_value", "max_value"],
                include_pixels=pixels,
                warn_on_modified_catalog=False,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logging.debug(
                "Could not read partition statistics for catalog %s", self.catalog.name, exc_info=True
            )
            return None

    # pylint: disable=protected-access
    @property
    def _show_statistics(self) -> bool:
        """Whether the user asked for statistics and the current view still preserves them."""
        return (
            self.catalog.loading_config is not None
            and self.catalog.loading_config.show_statistics
            and self.catalog._operation.pixel_stats_preserved
            and len(self.catalog.get_healpix_pixels()) > 0
        )

    @property
    def _divisions(self) -> pd.Index:
        pixels = self.catalog.get_ordered_healpix_pixels()
        name = f"npartitions={len(pixels)}"
        labels: list[HealpixPixel | str] = list(pixels)
        if len(pixels) == 0:
            labels = ["Empty Catalog"]
        elif len(pixels) > _REPR_MAX_ROWS - 1:
            # First pixel and last two pixels, with ellipsis in between
            labels = [pixels[0], "...", *pixels[-2:]]
        return pd.Index(labels, name=name)


def _repr_data_min_max(meta, index, pixel_stats) -> pd.DataFrame:
    """Build repr DataFrame with a dtype header row and per-pixel '{min}..{max}' for every pixel"""
    dtype_index = pd.Index([index.name] + list(index), name=None)
    stats_by_pixel = {pixel: pixel_stats.loc[pixel] for pixel in pixel_stats.index}
    series_list = []
    for col_name, s in meta.items():
        min_col = f"{col_name}: min_value"
        max_col = f"{col_name}: max_value"
        has_stats = min_col in pixel_stats.columns and max_col in pixel_stats.columns
        pixel_values = (
            _make_pixel_range(index, stats_by_pixel, min_col, max_col, s.dtype)
            if has_stats
            else ["..."] * len(index)
        )
        values = [str(s.dtype)] + pixel_values
        series_list.append(pd.Series(values, index=dtype_index, name=col_name))
    return pd.concat(series_list, axis=1)


def _make_pixel_range(index, stats_by_pixel, min_col, max_col, dtype) -> list[str]:
    """Make a '{min}..{max}' string for each pixel in the index"""
    formatter = _get_formatter(dtype)
    values = []
    for pixel in index:
        row = stats_by_pixel.get(pixel)
        if row is None or pd.isna(row[min_col]) or pd.isna(row[max_col]):
            values.append("...")
            continue
        min_str, max_str = formatter(row[min_col]), formatter(row[max_col])
        values.append(min_str if min_str == max_str else f"{min_str}..{max_str}")
    return values


def _get_formatter(dtype):
    """Return a function to format a given scalar of dtype"""
    if pd.api.types.is_bool_dtype(dtype):
        return lambda v: str(bool(v))

    if pd.api.types.is_float_dtype(dtype):
        return lambda v: f"{float(v):.4g}"

    if pd.api.types.is_integer_dtype(dtype):

        def fmt_int(v):
            s = int_comma(int(v))
            return s if len(s) <= _MAX_STR_WIDTH else f"{int(v):.4g}"

        return fmt_int

    def fmt_str(v):
        s = str(v)
        return s if len(s) <= _MAX_STR_WIDTH else s[: _MAX_STR_WIDTH - 1] + "»"

    return fmt_str
