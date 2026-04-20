from __future__ import annotations

import pandas as pd

_MAX_STR_WIDTH = 10


def _repr_data_min_max(meta, index, pixel_stats) -> pd.DataFrame:
    """Build repr DataFrame with a dtype header row and per-pixel min..max for every pixel."""
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
    """Make a 'min..max' string for each pixel in the index."""
    formatter = _get_formatter(dtype)
    values = []
    for pixel in index:
        row = stats_by_pixel.get(pixel)
        if row is None or row[min_col] is None or row[max_col] is None:
            values.append("...")
            continue
        min_str, max_str = formatter(row[min_col]), formatter(row[max_col])
        values.append(min_str if min_str == max_str else f"{min_str}..{max_str}")
    return values


def _get_formatter(dtype):
    """Return a scalar formatting function for the given dtype, resolved once per column."""
    if pd.api.types.is_bool_dtype(dtype):
        return lambda v: str(bool(v))
    if pd.api.types.is_integer_dtype(dtype):
        return lambda v: f"{int(v):.4g}" if abs(int(v)) >= 1_000_000 else str(int(v))
    if pd.api.types.is_float_dtype(dtype):
        return lambda v: f"{float(v):.4g}"

    def fmt_str(v):
        s = str(v)
        return s if len(s) <= _MAX_STR_WIDTH else s[: _MAX_STR_WIDTH - 1] + "…"

    return fmt_str
