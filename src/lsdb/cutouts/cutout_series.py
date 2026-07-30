"""CutoutSeries and the ``.cutout`` series accessor.

The accessor holds all cutout behavior and works on any series of cutout dtype
(including plain ``pd.Series``). :class:`CutoutSeries` is stateless sugar over
the accessor, and is registered with nested-pandas so that ``df["cutouts"]``
on a ``NestedFrame`` returns it directly.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd
from nested_pandas import register_series_class

from lsdb.cutouts.cutout_array import CutoutArray, CutoutDtype
from lsdb.cutouts.image_store import ImageStore

__all__ = ["CutoutSeries", "CutoutAccessor"]


@pd.api.extensions.register_series_accessor("cutout")
class CutoutAccessor:
    """Accessor for cutout columns: ``series.cutout.<method>``.

    Raises AttributeError when used on a series that is not of cutout dtype.
    """

    def __init__(self, series: pd.Series):
        if not isinstance(series.dtype, CutoutDtype):
            raise AttributeError("Can only use .cutout accessor with a 'cutout' dtype series.")
        self._series = series

    @property
    def _array(self) -> CutoutArray:
        return self._series.array  # type: ignore[return-value]

    @property
    def store(self) -> ImageStore | None:
        """The image store attached to this column, or None."""
        return self._array.store

    def with_store(self, store: ImageStore | None) -> pd.Series:
        """Return a new series with the given image store attached.

        Parameters
        ----------
        store : ImageStore or None
            The store resolving this column's image ids to pixels.

        Returns
        -------
        CutoutSeries
            A new series sharing this series' descriptors.
        """
        array = self._array.with_store(store)
        return CutoutSeries(array, index=self._series.index, name=self._series.name)

    def to_images(self) -> list[np.ndarray | None]:
        """Render every cutout as a numpy array view into the stored images.

        Returns
        -------
        list of np.ndarray or None
            One 2D array per row (a zero-copy view into the source image);
            None for missing cutouts.
        """
        return [None if ref is pd.NA else ref.data for ref in self._array]

    def to_image_stack(self) -> np.ndarray:
        """Render all cutouts into a single stacked ``(n, height, width)`` array.

        All cutouts must have the same shape and no row may be missing.
        Unlike :meth:`to_images`, the result is a contiguous copy.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(series), height, width)``.
        """
        images = self.to_images()
        if any(image is None for image in images):
            raise ValueError("Cannot stack: series contains missing cutouts")
        shapes = {image.shape for image in images}  # type: ignore[union-attr]
        if len(shapes) > 1:
            raise ValueError(f"Cannot stack cutouts of differing shapes: {sorted(shapes)}")
        return np.stack(images)  # type: ignore[arg-type]

    def to_cutout2d(self, copy: bool = False) -> list:
        """Render every cutout as an `astropy.nddata.Cutout2D`.

        Each Cutout2D carries a WCS adjusted to the cutout frame when the
        image store provides one for the source image.

        Parameters
        ----------
        copy : bool
            If False (default), cutout ``.data`` are zero-copy views into the
            stored images; note that a view keeps its whole source image
            alive. If True, pixels are copied so the cutouts are detached
            from the image store.

        Returns
        -------
        list of astropy.nddata.Cutout2D or None
            One Cutout2D per row; None for missing cutouts.
        """
        return [None if ref is pd.NA else ref.to_cutout2d(copy=copy) for ref in self._array]

    def to_descriptor_frame(self) -> pd.DataFrame:
        """Return the cutout descriptors as a plain DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns ``image_id``, ``x0``, ``y0``, ``width``, ``height``,
            indexed like this series.
        """
        frame = self._array.to_descriptor_frame()
        frame.index = self._series.index
        return frame


def cutout_only(func):
    """Restrict a CutoutSeries method to series of cutout dtype."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.dtype, CutoutDtype):
            raise TypeError(f"'{func.__name__}' can only be used with a 'cutout' dtype series.")
        return func(self, *args, **kwargs)

    return wrapper


class CutoutSeries(pd.Series):
    """A Series of image cutouts.

    Stateless view over a cutout-dtype series: all state lives in the
    underlying :class:`CutoutArray`, so wrapping and unwrapping to plain
    ``pd.Series`` is lossless. Returned automatically by ``NestedFrame``
    column access for cutout columns.
    """

    @property
    @cutout_only
    def store(self) -> ImageStore | None:
        """The image store attached to this column, or None."""
        return self.cutout.store

    @cutout_only
    def with_store(self, store: ImageStore | None) -> CutoutSeries:
        """Return a new series with the given image store attached."""
        return self.cutout.with_store(store)

    @cutout_only
    def to_images(self) -> list[np.ndarray | None]:
        """Render every cutout as a numpy array view into the stored images."""
        return self.cutout.to_images()

    @cutout_only
    def to_image_stack(self) -> np.ndarray:
        """Render all cutouts into a single stacked ``(n, height, width)`` array."""
        return self.cutout.to_image_stack()

    @cutout_only
    def to_cutout2d(self, copy: bool = False) -> list:
        """Render every cutout as an `astropy.nddata.Cutout2D`.

        With the default ``copy=False`` the cutout data are zero-copy views
        into the stored images; pass ``copy=True`` for detached copies.
        """
        return self.cutout.to_cutout2d(copy=copy)

    @cutout_only
    def to_descriptor_frame(self) -> pd.DataFrame:
        """Return the cutout descriptors as a plain DataFrame."""
        return self.cutout.to_descriptor_frame()

    def __repr__(self) -> str:
        if not isinstance(self.dtype, CutoutDtype):
            return super().__repr__()
        from lsdb.cutouts import display  # pylint: disable=import-outside-toplevel,cyclic-import

        return display.series_repr(self)

    def _repr_html_(self) -> str | None:
        """HTML repr with image thumbnails for the first cutouts (used by notebooks)."""
        if not isinstance(self.dtype, CutoutDtype):
            return None
        from lsdb.cutouts import display  # pylint: disable=import-outside-toplevel,cyclic-import

        return display.series_html(self)


register_series_class(CutoutDtype, CutoutSeries)
