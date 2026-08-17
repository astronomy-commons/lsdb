"""Image columns: tensor-backed image series.

An image column is a tensor column whose cells are image pixel arrays
(2-d cutout stamps, or 3-d stacks with a leading plane/band axis).
:class:`ImageDtype` subclasses nested-pandas ``TensorDtype`` and
:class:`ImageSeries` subclasses ``TensorSeries``; the nested-pandas
series-class registry dispatches on the dtype's MRO, so image columns come
back as ``ImageSeries`` while plain tensor columns stay ``TensorSeries``.

This is the tensor-backed (materialized pixels) entry point of the image
column design. A store-backed variant — descriptor rows resolved against a
shared image store — can later be added as a sibling series class registered
for its own dtype, sharing the ``ImageSeries`` API.
"""

from __future__ import annotations

import numpy as np
from nested_pandas import TensorDtype, TensorSeries, register_series_class
from pandas.api.extensions import register_extension_dtype

__all__ = ["ImageDtype", "ImageSeries"]


@register_extension_dtype
class ImageDtype(TensorDtype):
    """Data type for columns of image pixel arrays.

    Behaves exactly like ``TensorDtype`` (fixed shape via ``shape``, ragged
    via ``ndim``), but marks the column as image data so it is returned as an
    :class:`ImageSeries`. The dtype string uses the ``image`` prefix, e.g.
    ``"image[float, (25, 25)]"``.

    Note that serialization currently keeps only the tensor identity: an
    image column written to parquet is read back as a plain tensor column.
    """

    _name_prefix = "image"


class ImageSeries(TensorSeries):
    """A Series of image pixel arrays, one numpy array per row."""

    def to_image_stack(self, na_value=np.nan) -> np.ndarray:
        """Convert to a single (n, height, width, ...) numpy pixel block.

        Missing rows are filled with ``na_value``. Only available when all
        rows share one shape (always true for fixed-shape image columns).
        """
        return self.to_stack(na_value=na_value)


register_series_class(ImageDtype, ImageSeries)
