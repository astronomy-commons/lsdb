"""Text and HTML rendering for cutout columns.

Text reprs never touch pixel data (rendering may require remote fetches for
lazy image stores); cutouts are shown as descriptor text with a placeholder
star. HTML reprs render the first :data:`MAX_RENDERED` cutouts as inline PNG
thumbnails and show the rest as a text placeholder.

Importing :mod:`lsdb.cutouts` registers an IPython display formatter so that
``NestedFrame`` objects containing cutout columns show thumbnails in notebooks
without any changes to nested-pandas.
"""

from __future__ import annotations

import base64
import html as html_module
import io

import nested_pandas as npd
import numpy as np
import pandas as pd

from lsdb.cutouts.cutout_array import CutoutDtype, CutoutRef

__all__ = ["MAX_RENDERED", "render_png_base64", "series_repr", "series_html", "nestedframe_html"]

# Number of cutouts rendered as actual images in HTML previews.
MAX_RENDERED = 10

_THUMBNAIL_STYLE = "width:64px;image-rendering:pixelated;"
_PLACEHOLDER_HTML = '<span style="color:#888;">&lt;not rendered in preview&gt;</span>'

_ASCII_STAR = (
    r"     \ | /   ",
    r"   --  *  -- ",
    r"     / | \   ",
)


def _descriptor_text(ref: CutoutRef) -> str:
    return f"{ref.image_id}[{ref.y0}:{ref.y0 + ref.height}, {ref.x0}:{ref.x0 + ref.width}]"


def render_png_base64(data: np.ndarray) -> str | None:
    """Render a 2D array as a base64-encoded grayscale PNG.

    Pixel values are clipped to the 1st-99th percentile before rendering.

    Parameters
    ----------
    data : np.ndarray
        2D pixel array.

    Returns
    -------
    str or None
        Base64-encoded PNG bytes, or None if matplotlib is unavailable.
    """
    try:
        from matplotlib import image as mpl_image  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    buffer = io.BytesIO()
    mpl_image.imsave(buffer, data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", format="png")
    return base64.b64encode(buffer.getvalue()).decode()


def series_repr(series: pd.Series) -> str:
    """Text repr for a cutout series: descriptors and a placeholder star, no pixel access.

    Parameters
    ----------
    series : pd.Series
        A series of cutout dtype.

    Returns
    -------
    str
    """
    array = series.array
    store = array.store  # type: ignore[union-attr]
    lines = []
    for position in range(min(len(series), MAX_RENDERED)):
        ref = array[position]
        label = f"[{series.index[position]}]"
        if ref is pd.NA:
            lines.append(f"{label}  <NA>")
            continue
        lines.append(f"{label}  {_descriptor_text(ref)}  ({ref.height} x {ref.width})")
        lines.extend(_ASCII_STAR)
    if len(series) > MAX_RENDERED:
        lines.append(f"... {len(series) - MAX_RENDERED} more cutouts <not rendered in preview>")
    store_name = type(store).__name__ if store is not None else "None"
    lines.append(f"Length: {len(series)}, dtype: {series.dtype}, store: {store_name}")
    return "\n".join(lines)


def _cell_html(ref, rendered: bool) -> str:
    """HTML for a single cutout cell: a thumbnail, a placeholder, or descriptor text."""
    if ref is pd.NA or ref is None:
        return "&lt;NA&gt;"
    descriptor = html_module.escape(_descriptor_text(ref), quote=True)
    if not rendered:
        return _PLACEHOLDER_HTML
    if ref.store is None:
        return f'{descriptor} <span style="color:#888;">(no image store attached)</span>'
    png = render_png_base64(ref.data)
    if png is None:  # matplotlib unavailable
        return descriptor
    return f'<img src="data:image/png;base64,{png}" style="{_THUMBNAIL_STYLE}" title="{descriptor}"/>'


def series_html(series: pd.Series) -> str:
    """HTML repr for a cutout series: thumbnails for the first cutouts.

    Parameters
    ----------
    series : pd.Series
        A series of cutout dtype.

    Returns
    -------
    str
    """
    rows = []
    for position in range(len(series)):
        ref = series.array[position]
        preview = _cell_html(ref, rendered=position < MAX_RENDERED)
        descriptor = "" if ref is pd.NA else html_module.escape(_descriptor_text(ref))
        rows.append(
            f"<tr><th>{html_module.escape(str(series.index[position]))}</th>"
            f"<td>{preview}</td><td>{descriptor}</td></tr>"
        )
    name = html_module.escape(str(series.name)) if series.name is not None else ""
    header = f"<tr><th></th><th>{name}</th><th></th></tr>"
    footer = f"<p>Length: {len(series)}, dtype: {series.dtype}</p>"
    return f"<table>{header}{''.join(rows)}</table>{footer}"


def ref_html(ref: CutoutRef) -> str | None:
    """Notebook HTML for a single cutout: a larger thumbnail with a caption.

    Parameters
    ----------
    ref : CutoutRef
        The cutout to render.

    Returns
    -------
    str or None
        HTML, or None when no store is attached or matplotlib is unavailable
        (the caller then falls back to a text repr).
    """
    if ref.store is None:
        return None
    png = render_png_base64(ref.data)
    if png is None:
        return None
    descriptor = html_module.escape(_descriptor_text(ref), quote=True)
    return (
        f'<figure style="margin:0;">'
        f'<img src="data:image/png;base64,{png}" style="width:128px;image-rendering:pixelated;"/>'
        f'<figcaption style="font-family:monospace;font-size:0.8em;">'
        f"{descriptor} ({ref.height} x {ref.width})</figcaption></figure>"
    )


def nestedframe_html(frame: pd.DataFrame, max_rows: int = 10) -> str:
    """HTML repr for a frame, rendering cutout columns as thumbnails.

    Frames without cutout columns fall back to the default pandas rendering.
    Non-cutout columns are formatted through pandas' own machinery (so
    extension dtypes like nested columns render as they normally would) and
    escaped, since ``escape=False`` is needed for the thumbnail ``<img>`` tags.

    Parameters
    ----------
    frame : pd.DataFrame
        The frame to render.
    max_rows : int
        Maximum number of rows to display.

    Returns
    -------
    str
    """
    # pylint: disable-next=import-outside-toplevel
    from pandas.io.formats.format import format_array  # type: ignore[attr-defined]

    cutout_columns = [name for name in frame.columns if isinstance(frame.dtypes[name], CutoutDtype)]
    if not cutout_columns:
        return frame._repr_html_()  # pylint: disable=protected-access

    # Pre-render every cell to an HTML string: thumbnails for the first
    # MAX_RENDERED cutouts, pandas-formatted (then escaped) text for the rest
    head = pd.DataFrame(frame.head(max_rows))
    rendered = {}
    for name in head.columns:
        if name in cutout_columns:
            array = head[name].array
            rendered[name] = [
                _cell_html(array[position], rendered=position < MAX_RENDERED) for position in range(len(head))
            ]
        else:
            strings = format_array(head[name].array, None)
            rendered[name] = [html_module.escape(value.strip()) for value in strings]
    cells = pd.DataFrame(rendered, index=head.index)

    # max_colwidth would truncate the base64 image payloads mid-string.
    with pd.option_context("display.max_colwidth", None):
        html = cells.to_html(escape=False, notebook=True)
    if len(frame) > max_rows:
        html += f"<p>... {len(frame) - max_rows} more rows</p>"
    return html


def register_ipython_formatter() -> None:
    """Register the NestedFrame HTML formatter with IPython, if running under it.

    Called on import of :mod:`lsdb.cutouts`. No-op outside IPython or when
    IPython is not installed.
    """
    try:
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    ipython = get_ipython()
    if ipython is None or getattr(ipython, "display_formatter", None) is None:
        return
    html_formatter = ipython.display_formatter.formatters["text/html"]
    html_formatter.for_type(npd.NestedFrame, nestedframe_html)
