"""The CutoutArray extension array, its dtype, and its scalar type.

A cutout is described by an image id and a pixel bounding box into that image.
Columns of ``CutoutDtype`` store these per-row descriptors as an Arrow struct
array, plus one shared :class:`ImageStore` reference for the whole array.
Pixels are never stored per row: rendering produces zero-copy numpy views into
the store's images, so overlapping cutouts share memory.

Only descriptors survive Arrow serialization (``__arrow_array__``); the store is
an in-memory attachment, re-attached where pixels are needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from pandas.core.indexers import check_array_indexer

from lsdb.cutouts.image_store import ImageStore, merge_stores

__all__ = ["CutoutArray", "CutoutDtype", "CutoutRef", "CUTOUT_ARROW_TYPE"]

# Arrow storage type for cutout descriptors (pixels are never stored per row).
CUTOUT_ARROW_TYPE = pa.struct(
    [
        pa.field("image_id", pa.string()),
        pa.field("x0", pa.int32()),
        pa.field("y0", pa.int32()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
    ]
)


@dataclass(frozen=True)
class CutoutRef:
    """A reference to a rectangular region of an image.

    Rendering (``.data``, ``.to_cutout2d()``) requires an attached image store;
    the descriptor fields alone are cheap, serializable metadata.

    Parameters
    ----------
    image_id : str
        Identifier of the source image in the image store.
    x0, y0 : int
        Pixel coordinates of the lower-left corner of the cutout in the
        source image (numpy convention: ``image[y0:y0+height, x0:x0+width]``).
    width, height : int
        Extent of the cutout in pixels.
    store : ImageStore, optional
        The store that can resolve ``image_id`` to pixels. Excluded from
        equality and repr.
    """

    image_id: str
    x0: int
    y0: int
    width: int
    height: int
    store: ImageStore | None = field(default=None, compare=False, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        """The (height, width) shape of the cutout."""
        return (self.height, self.width)

    @property
    def data(self) -> np.ndarray:
        """The cutout pixels.

        Reads through ``store.get_region``: a zero-copy view for in-memory
        stores; for file-backed stores this may read only the tiles/chunks
        the cutout intersects (see ``CatalogImageStore`` read modes).

        Raises
        ------
        ValueError
            If no image store is attached.
        """
        if self.store is None:
            raise ValueError(
                f"Cutout of image '{self.image_id}' has no image store attached; "
                "use `.cutout.with_store(store)` on the series to attach one."
            )
        return self.store.get_region(
            self.image_id, self.y0, self.y0 + self.height, self.x0, self.x0 + self.width
        )

    def to_cutout2d(self, copy: bool = False):
        """Render as an `astropy.nddata.Cutout2D`, with WCS if the store provides one.

        Note that ``Cutout2D`` is constructed against the parent image, so
        this loads the full image from the store (unlike ``.data``, which
        can read only the cutout's region).

        Parameters
        ----------
        copy : bool
            If False (default), the cutout ``.data`` is a zero-copy view into
            the stored image; note that a view keeps the whole source image
            alive. If True, the pixels are copied so the cutout is detached
            from the image store.

        Returns
        -------
        astropy.nddata.Cutout2D
            The rendered cutout, with ``.wcs`` (if present) adjusted to the
            cutout frame.
        """
        from astropy.nddata import Cutout2D  # pylint: disable=import-outside-toplevel

        if self.store is None:
            raise ValueError(
                f"Cutout of image '{self.image_id}' has no image store attached; "
                "use `.cutout.with_store(store)` on the series to attach one."
            )
        image = self.store.get_image(self.image_id)
        wcs = self.store.get_wcs(self.image_id)
        position = (self.x0 + (self.width - 1) / 2, self.y0 + (self.height - 1) / 2)
        return Cutout2D(
            image, position=position, size=(self.height, self.width), wcs=wcs, mode="partial", copy=copy
        )

    def _repr_html_(self) -> str | None:
        """Notebook display: the rendered cutout image with a descriptor caption.

        Returns None (falling back to the dataclass repr) when no store is
        attached or the image cannot be rendered.
        """
        from lsdb.cutouts import display  # pylint: disable=import-outside-toplevel,cyclic-import

        return display.ref_html(self)


@register_extension_dtype
class CutoutDtype(ExtensionDtype):
    """Pandas extension dtype for image cutout columns.

    Columns of this dtype store per-row cutout descriptors (image id and pixel
    bounding box) in Arrow, plus a single shared :class:`ImageStore` on the
    array that resolves image ids to pixels at render time.
    """

    name = "cutout"
    type = CutoutRef
    kind = "O"
    na_value = pd.NA

    @classmethod
    def construct_from_string(cls, string: str) -> CutoutDtype:
        """Construct this dtype from the string ``"cutout"``.

        Parameters
        ----------
        string : str
            The dtype string.

        Returns
        -------
        CutoutDtype
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CutoutDtype' from '{string}'")
        return cls()

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype.

        Returns
        -------
        type[CutoutArray]
        """
        return CutoutArray

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """Construct a CutoutArray (without a store) from an Arrow array.

        Parameters
        ----------
        array : pa.Array or pa.ChunkedArray
            Struct array of cutout descriptors.

        Returns
        -------
        CutoutArray
        """
        return CutoutArray(array, store=None)


_FIELD_NAMES = ("image_id", "x0", "y0", "width", "height")


class CutoutArray(ExtensionArray):
    """Extension array of image cutout descriptors with a shared image store.

    Parameters
    ----------
    values : pa.Array or pa.ChunkedArray
        Struct array of descriptors with fields ``image_id`` (string) and
        ``x0``, ``y0``, ``width``, ``height`` (integers). Null entries
        represent missing cutouts.
    store : ImageStore, optional
        Store resolving image ids to pixel arrays. May be None (descriptors
        only); rendering then raises until a store is attached.

    Examples
    --------
    >>> import numpy as np
    >>> from lsdb.cutouts import CutoutArray, InMemoryImageStore
    >>> store = InMemoryImageStore({"img1": np.arange(100.0).reshape(10, 10)})
    >>> array = CutoutArray.from_arrays(
    ...     image_id=["img1", "img1"], x0=[0, 1], y0=[0, 1], width=[3, 3], height=[3, 3], store=store
    ... )
    >>> array[0].data.shape
    (3, 3)
    """

    def __init__(self, values: pa.Array | pa.ChunkedArray, store: ImageStore | None = None):
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if not pa.types.is_struct(values.type):
            raise TypeError(f"CutoutArray expects an Arrow struct array, got {values.type}")
        if values.type != CUTOUT_ARROW_TYPE:
            values = values.cast(CUTOUT_ARROW_TYPE)
        self._pa = values
        self._store = store

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_arrays(
        cls,
        image_id: Sequence[str],
        x0: Sequence[int],
        y0: Sequence[int],
        width: Sequence[int],
        height: Sequence[int],
        store: ImageStore | None = None,
    ) -> CutoutArray:
        """Build a CutoutArray from per-field sequences.

        Parameters
        ----------
        image_id : sequence of str
            Image identifiers, one per cutout.
        x0 : sequence of int
            Pixel column of the lower-left corner of each cutout.
        y0 : sequence of int
            Pixel row of the lower-left corner of each cutout.
        width : sequence of int
            Pixel width of each cutout.
        height : sequence of int
            Pixel height of each cutout.
        store : ImageStore, optional
            Store resolving the image ids to pixels.

        Returns
        -------
        CutoutArray
        """
        struct = pa.StructArray.from_arrays(
            [
                pa.array(image_id, type=pa.string()),
                pa.array(x0, type=pa.int32()),
                pa.array(y0, type=pa.int32()),
                pa.array(width, type=pa.int32()),
                pa.array(height, type=pa.int32()),
            ],
            names=list(_FIELD_NAMES),
        )
        return cls(struct, store=store)

    @classmethod
    def _from_sequence(cls, scalars: Iterable, *, dtype=None, copy: bool = False) -> CutoutArray:
        rows: list[dict | None] = []
        stores: list[ImageStore | None] = []
        for scalar in scalars:
            if isinstance(scalar, CutoutRef):
                rows.append(
                    {
                        "image_id": scalar.image_id,
                        "x0": scalar.x0,
                        "y0": scalar.y0,
                        "width": scalar.width,
                        "height": scalar.height,
                    }
                )
                stores.append(scalar.store)
            elif isinstance(scalar, dict):
                rows.append(scalar)
            elif scalar is None or scalar is pd.NA or (isinstance(scalar, float) and np.isnan(scalar)):
                rows.append(None)
            else:
                raise TypeError(f"Cannot build a CutoutArray from scalar of type {type(scalar)}")
        struct = pa.array(rows, type=CUTOUT_ARROW_TYPE)
        return cls(struct, store=merge_stores(stores))

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError("CutoutArray does not support factorization")

    # ------------------------------------------------------------------ #
    # ExtensionArray interface
    # ------------------------------------------------------------------ #

    @property
    def dtype(self) -> CutoutDtype:
        return CutoutDtype()

    @property
    def nbytes(self) -> int:
        return self._pa.nbytes

    def __len__(self) -> int:
        return len(self._pa)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            scalar = self._pa[int(item)]
            if not scalar.is_valid:
                return pd.NA
            values = scalar.as_py()
            return CutoutRef(
                image_id=values["image_id"],
                x0=values["x0"],
                y0=values["y0"],
                width=values["width"],
                height=values["height"],
                store=self._store,
            )
        item = check_array_indexer(self, item)
        if isinstance(item, slice):
            return type(self)(self._pa[item], store=self._store)
        item = np.asarray(item)
        if item.dtype == bool:
            return type(self)(self._pa.filter(pa.array(item)), store=self._store)
        return self.take(item)

    def __setitem__(self, key, value):
        raise NotImplementedError("CutoutArray is immutable; build a new array instead")

    def __eq__(self, other):
        if isinstance(other, CutoutArray):
            result = np.zeros(len(self), dtype=bool)
            for i in range(min(len(self), len(other))):
                result[i] = self[i] == other[i]
            return result
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return ~result

    def isna(self) -> np.ndarray:
        return np.asarray(self._pa.is_null())

    def take(self, indices, *, allow_fill: bool = False, fill_value=None) -> CutoutArray:
        indices = np.asarray(indices, dtype=np.int64)
        if allow_fill:
            if fill_value is not None and fill_value is not pd.NA:
                raise NotImplementedError("CutoutArray only supports NA as a fill value")
            if len(self) == 0 and (indices >= 0).any():
                raise IndexError("cannot take from an empty CutoutArray with positive indices")
            if (indices < -1).any():
                raise ValueError("indices must be >= -1 when allow_fill is True")
            mask = indices == -1
            arrow_indices = pa.array(np.where(mask, 0, indices), mask=mask)
        else:
            indices = np.where(indices < 0, indices + len(self), indices)
            if len(indices) and ((indices < 0).any() or (indices >= len(self)).any()):
                raise IndexError("index out of bounds for CutoutArray take")
            arrow_indices = pa.array(indices)
        return type(self)(self._pa.take(arrow_indices), store=self._store)

    def copy(self) -> CutoutArray:
        # Arrow buffers are immutable; sharing them is safe.
        return type(self)(self._pa, store=self._store)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[CutoutArray]) -> CutoutArray:
        arrays = [array._pa for array in to_concat]  # pylint: disable=protected-access
        store = merge_stores([array._store for array in to_concat])
        return cls(pa.concat_arrays(arrays), store=store)

    def _formatter(self, boxed: bool = False):
        def fmt(value):
            if value is pd.NA:
                return str(value)
            return (
                f"{value.image_id}"
                f"[{value.y0}:{value.y0 + value.height}, {value.x0}:{value.x0 + value.width}]"
            )

        return fmt

    # ------------------------------------------------------------------ #
    # Arrow interoperability
    # ------------------------------------------------------------------ #

    def __arrow_array__(self, type=None):  # pylint: disable=redefined-builtin
        """Convert to an Arrow array of descriptors. The image store is dropped."""
        if type is not None and type != CUTOUT_ARROW_TYPE:
            return self._pa.cast(type)
        return self._pa

    # ------------------------------------------------------------------ #
    # Cutout-specific API
    # ------------------------------------------------------------------ #

    @property
    def store(self) -> ImageStore | None:
        """The attached image store, or None if descriptors-only."""
        return self._store

    def with_store(self, store: ImageStore | None) -> CutoutArray:
        """Return a new array with the given image store attached.

        Parameters
        ----------
        store : ImageStore or None
            The store to attach (None detaches).

        Returns
        -------
        CutoutArray
            A new array sharing this array's descriptors.
        """
        return type(self)(self._pa, store=store)

    def field(self, name: str) -> np.ndarray:
        """Return one descriptor field as a numpy array.

        Parameters
        ----------
        name : str
            One of ``image_id``, ``x0``, ``y0``, ``width``, ``height``.

        Returns
        -------
        np.ndarray
        """
        return self._pa.field(name).to_numpy(zero_copy_only=False)

    def to_descriptor_frame(self) -> pd.DataFrame:
        """Return the descriptors as a plain DataFrame (one column per field).

        Returns
        -------
        pd.DataFrame
            Columns ``image_id``, ``x0``, ``y0``, ``width``, ``height``.
        """
        return pd.DataFrame({name: self.field(name) for name in _FIELD_NAMES})
