"""Image stores: sources of pixel data that cutout descriptors reference by image id.

A :class:`CutoutArray` holds lightweight descriptors (image id and pixel bounding box)
and an optional ``ImageStore`` that can resolve those ids to actual pixel arrays.
Descriptors are serializable and travel through Dask graphs; stores are attached
where pixels are actually needed (inside a rendering ``map_partitions`` or
client-side after ``compute()``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from astropy.wcs import WCS


class ImageStore(ABC):
    """Resolves image ids to pixel arrays (and optionally WCS objects)."""

    @abstractmethod
    def get_image(self, image_id: str) -> np.ndarray:
        """Return the full pixel array for an image id.

        Parameters
        ----------
        image_id : str
            The identifier of the image to fetch.

        Returns
        -------
        np.ndarray
            2D pixel array for the image.
        """

    # pylint: disable-next=unused-argument, redundant-returns-doc
    def get_wcs(self, image_id: str) -> WCS | None:
        """Return the WCS for an image id, or None if unavailable.

        Parameters
        ----------
        image_id : str
            The identifier of the image.

        Returns
        -------
        astropy.wcs.WCS or None
            The image WCS, or None if the store has no WCS information.
        """
        return None

    @abstractmethod
    def __contains__(self, image_id: str) -> bool: ...

    def get_region(self, image_id: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Return one rectangular region of an image.

        The base implementation slices the full image (a zero-copy view for
        in-memory stores); stores backed by chunked/tiled files may override
        it to read only the bytes the region needs. Bounds are clamped to the
        image, so the result may be smaller than requested at image edges.

        Parameters
        ----------
        image_id : str
            The identifier of the image.
        y0 : int
            First row of the region.
        y1 : int
            End row (exclusive) of the region.
        x0 : int
            First column of the region.
        x1 : int
            End column (exclusive) of the region.

        Returns
        -------
        np.ndarray
            2D pixel array of the (clamped) region.
        """
        image = self.get_image(image_id)
        return image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]

    def plan_reads(self, image_ids) -> None:
        """Announce the images an upcoming batch of renders will touch.

        Called by batch operations (``to_images``, ``to_image_stack``) with
        one entry per cutout, before any pixels are requested. Stores may use
        the counts to choose their read strategy upfront (e.g. load an image
        in full immediately when many cutouts will need it, instead of
        discovering that one region at a time). The base implementation does
        nothing.

        Parameters
        ----------
        image_ids : iterable of str
            The image id of every cutout about to be rendered (with
            repetitions; multiplicity is the signal).
        """


class InMemoryImageStore(ImageStore):
    """An image store backed by in-memory numpy arrays.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from image id to 2D pixel array.
    wcs : dict[str, astropy.wcs.WCS], optional
        Mapping from image id to WCS. Ids without an entry return None.

    Examples
    --------
    >>> import numpy as np
    >>> from lsdb.cutouts import InMemoryImageStore
    >>> store = InMemoryImageStore({"img1": np.zeros((100, 100))})
    >>> "img1" in store
    True
    >>> store.get_image("img1").shape
    (100, 100)
    """

    def __init__(self, images: dict[str, np.ndarray], wcs: dict[str, WCS] | None = None):
        self.images = dict(images)
        self.wcs = dict(wcs) if wcs is not None else {}

    def get_image(self, image_id: str) -> np.ndarray:
        return self.images[image_id]

    def get_wcs(self, image_id: str) -> WCS | None:
        return self.wcs.get(image_id)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self.images


class ChainImageStore(ImageStore):
    """An image store that resolves ids by querying a sequence of stores in order.

    Used when concatenating cutout arrays whose stores differ: the result
    references all of them without copying pixel data.

    Parameters
    ----------
    stores : list[ImageStore]
        The stores to query, in priority order.
    """

    def __init__(self, stores: list[ImageStore]):
        self.stores = stores

    def get_image(self, image_id: str) -> np.ndarray:
        for store in self.stores:
            if image_id in store:
                return store.get_image(image_id)
        raise KeyError(image_id)

    def get_wcs(self, image_id: str) -> WCS | None:
        for store in self.stores:
            if image_id in store:
                return store.get_wcs(image_id)
        return None

    def get_region(self, image_id: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        for store in self.stores:
            if image_id in store:
                return store.get_region(image_id, y0, y1, x0, x1)
        raise KeyError(image_id)

    def plan_reads(self, image_ids) -> None:
        for store in self.stores:
            store.plan_reads([image_id for image_id in image_ids if image_id in store])

    def __contains__(self, image_id: str) -> bool:
        return any(image_id in store for store in self.stores)


def merge_stores(stores: list[ImageStore | None]) -> ImageStore | None:
    """Merge the stores of multiple cutout arrays into a single store.

    Identical store objects are deduplicated; distinct stores are combined
    into a :class:`ChainImageStore` (flattening nested chains).

    Parameters
    ----------
    stores : list[ImageStore or None]
        Stores to merge; None entries are ignored.

    Returns
    -------
    ImageStore or None
        A single store resolving all ids, or None if no input has a store.
    """
    unique: list[ImageStore] = []
    for store in stores:
        if store is None:
            continue
        candidates = store.stores if isinstance(store, ChainImageStore) else [store]
        for candidate in candidates:
            if not any(candidate is seen for seen in unique):
                unique.append(candidate)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return ChainImageStore(unique)
