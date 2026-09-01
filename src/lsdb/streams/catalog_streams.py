from collections.abc import Iterator
from typing import Optional

import dask
import hats
import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
import pandas as pd
from dask.delayed import Delayed
from dask.distributed import Client, Future
from hats.inspection import read_skymap

from lsdb import Catalog


class _FakeFuture:
    """Duck-typed `Future` interface for a pre-computed value.

    Parameters
    ----------
    obj
        Value to hold
    """

    def __init__(self, obj):
        self.obj = obj

    def result(self) -> pd.DataFrame:
        """Return the held value."""
        return self.obj


class CatalogStream:
    """Stream partitons from a catalog

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).

    Parameters
    ----------
    catalog : lsdb.Catalog
        A catalog to iterate over.
    client : dask.distributed.Client or None
        Dask client for distributed computation. None means running
        in a synced way with `dask.compute()` instead of asynced with
        `client.compute()`.
    partitions_per_chunk : int
        Number of partitions to yield. It will be clipped to the total number
        of partitions. By default, one partition will be yielded at a time,
        however, if using a distributed client, it's recommended to set this to
        at least 2x the number of workers to allow proper parallelism.
    shuffle : bool
        Whether to shuffle the partition order before streaming. If False, the
        partitions will be streamed in their original order. True by default.
        Additionally, if `shuffle` is True, the rows within each partition will
        also be shuffled.
    seed : int
        Random seed to use for observation sampling, when shuffling partitions.

    Examples
    --------
    Consider a toy catalog, which contains 12 data partitions:

    >>> import lsdb
    >>> from lsdb.streams import CatalogStream
    >>> cat = lsdb.generate_catalog(500, 10, seed=1)
    >>> cat.npartitions
    12

    The following grabs 4 random partitions 5 times in a row, looping over the data as needed:

    >>> cat_stream = CatalogStream(catalog=cat, partitions_per_chunk=4, seed=1)
    >>> for chunk in cat_stream:
    ...     print(len(chunk))
    171
    154
    175
    """

    def __init__(
        self,
        catalog: Catalog,
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.catalog = catalog

        if not isinstance(catalog, Catalog):
            raise ValueError(f"The provided catalog input type {type(catalog)} is not a lsdb.Catalog object.")

        self.operation = catalog._operation  # pylint: disable=protected-access
        self._pixels = catalog._operation.healpix_pixels  # pylint: disable=protected-access

        self.client = client
        self.partitions_per_chunk = min(partitions_per_chunk, self.catalog.npartitions)
        self.shuffle = shuffle
        self.seed = seed

        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng((1 << 32, self.seed))

    def get_next_partitions(
        self, partitions_left: np.ndarray, rng: np.random.Generator  # pylint: disable=unused-argument
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the next set of partitions to iterate over."""
        # Chomp a subset of partitions when running once through the data
        return (
            partitions_left[: -self.partitions_per_chunk],
            partitions_left[-self.partitions_per_chunk :],
        )

    def submit_next_partitions(self, partitions: np.ndarray) -> Future | _FakeFuture:
        """Submit the next set of partitions for computation."""

        # Intended to be used with single partition builds
        def _to_delayed(operation, pixel):
            build = operation.build(pixels=[pixel])
            graph = build.graph
            key = build.pixel_to_key_map[pixel]
            return Delayed(key, graph)

        selected = [_to_delayed(self.operation, self._pixels[i]) for i in partitions]

        if len(selected) == 1:
            if self.client is None:
                return _FakeFuture(selected[0].compute())
            return self.client.compute(selected[0])

        combined = dask.delayed(pd.concat)(selected)
        if self.client is None:
            return _FakeFuture(combined.compute())
        return self.client.compute(combined)

    def __iter__(self) -> "CatalogIterator":
        """Return an iterator for this iterable."""
        # Split the RNG: create a new one for the iterator
        iterator_rng = self.rng.spawn(1)[0]
        return CatalogIterator(self, rng=iterator_rng)


class InfiniteStream(CatalogStream):
    """Stream continuously yielding random subsets of partitions from a catalog.

    The data is pre-fetched on the background, 'n_workers' number
    of partitions per time (derived from `client` object).

    Parameters
    ----------
    catalog : lsdb.Catalog
        A catalog to iterate over.
    client : dask.distributed.Client or None
        Dask client for distributed computation. None means running
        in a synced way with `dask.compute()` instead of asynced with
        `client.compute()`.
    partitions_per_chunk : int
        Number of partitions to yield. It will be clipped to the total number
        of partitions. Be mindful when setting this value larger than 1, as
        holding multiple partitions in memory at once will increase memory usage.
    seed : int
        Random seed to use for observation sampling.

    Examples
    --------
    Consider a toy catalog, which contains 12 data partitions:

    >>> import lsdb
    >>> from lsdb.streams import InfiniteStream
    >>> cat = lsdb.generate_catalog(500, 10, seed=1)
    >>> cat.npartitions
    12

    The following grabs 4 random partitions 5 times in a row, looping over the data as needed:

    >>> inf_stream = InfiniteStream(catalog=cat, partitions_per_chunk=4, seed=1)
    >>> cat_iter = iter(inf_stream)
    >>> for _ in range(5):
    ...     chunk = next(cat_iter)
    ...     print(len(chunk))
    157
    185
    165
    169
    185
    """

    def __init__(
        self,
        catalog: Catalog,
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            catalog=catalog,
            client=client,
            partitions_per_chunk=partitions_per_chunk,
            seed=seed,
        )

    def get_next_partitions(
        self, partitions_left: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the next set of partitions to iterate over."""
        return partitions_left, rng.choice(partitions_left, self.partitions_per_chunk, replace=False)


class CatalogIterator(Iterator[pd.DataFrame]):
    """Iterator yielding random subsets of partitions from a catalog."""

    def __init__(self, iterable: CatalogStream, rng: np.random.Generator) -> None:
        self.rng = rng  # Use the iterator's own RNG
        self.iterable = iterable
        self.partitions_left = self._get_initial_partitions()
        self._empty = False
        self.future: Optional[Future | _FakeFuture] = self.iterable.submit_next_partitions(
            self._get_next_partitions()
        )

    def _get_initial_partitions(self) -> np.ndarray:
        """Initialize the partitions left to iterate over."""
        if self.iterable.shuffle:
            return self.rng.permutation(self.iterable.catalog.npartitions)
        return np.arange(self.iterable.catalog.npartitions)

    def _get_next_partitions(self) -> np.ndarray:
        """Get the next set of partitions to process."""
        self.partitions_left, partitions = self.iterable.get_next_partitions(self.partitions_left, self.rng)
        return partitions

    def __iter__(self) -> "CatalogIterator":
        return self

    def __next__(self) -> pd.DataFrame:
        if self._empty or self.future is None:
            raise StopIteration("All partitions have been processed")

        result: pd.DataFrame = self.future.result()

        if self.iterable.shuffle:
            result = result.sample(frac=1, random_state=self.rng)

        if len(self.partitions_left) > 0:
            self.future = self.iterable.submit_next_partitions(self._get_next_partitions())
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        # Fail gracefully if the iterable is an InfiniteStream
        if isinstance(self.iterable, InfiniteStream):
            raise TypeError("Length is not defined for an InfiniteStream.")

        return int(np.ceil(len(self.partitions_left) / self.iterable.partitions_per_chunk))


class CrossMatchStream:
    def __init__(
        self,
        catalog: Catalog,
        *crossmatching_kwargs: dict[str, object],
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.catalog = catalog
        self.client = client
        self.partitions_per_chunk = partitions_per_chunk
        self.shuffle = shuffle
        self.seed = seed

        self.crossmatching_kwargs = []
        for kwargs in crossmatching_kwargs:
            new_kwargs = kwargs.copy()
            new_kwargs["suffixes"] = ("", "_" + kwargs["other"].name)
            new_kwargs["suffix_method"] = "all_columns"

        self.accumulative_meta = [catalog.meta]
        result_catalog = self.catalog
        for right_catalog, right_catalog_kwargs in crossmatching_kwargs:
            result_catalog = result_catalog.crossmatch(**right_catalog_kwargs).map_partitions(
                lambda df: df.drop(columns=["_dist_arcsec"])
            )
            self.accumulative_meta.append(result_catalog.meta)

        self._pixels = result_catalog._operation.healpix_pixels

        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng((1 << 32, self.seed))

        right_catalogs = [kwargs["other"] for kwargs in crossmatching_kwargs]
        self.mask_generator = CountMapForPixel(self.catalog, right_catalogs, count_fraction=0.5)

    def submit_next_partitions(self, partitions: np.ndarray) -> Future | _FakeFuture:
        """Submit the next set of partitions for computation."""

        # Intended to be used with single partition builds
        def _to_delayed(operation, pixel):
            build = operation.build(pixels=[pixel])
            graph = build.graph
            key = build.pixel_to_key_map[pixel]
            return Delayed(key, graph)

        selected = []

        for pixel_index in partitions:
            pixel = self._pixels[pixel_index]
            right_catalog_mask = self.mask_generator.get_pixel_catalog_mask(pixel, self.rng)

            def skipped_crossmatch(
                partition: npd.NestedFrame, *, meta_to_match: npd.NestedFrame
            ) -> npd.NestedFrame:
                old_n_columns = partition.shape[1]
                new_columns = meta_to_match.columns[old_n_columns:]
                for column in new_columns:
                    partition[column] = pd.Series(None, dtype=meta_to_match[column].dtype)
                return partition

            result_catalog = self.catalog
            for cross_match_kwargs, do_crossmatch, meta_to_match in zip(
                self.crossmatch_kwargs, right_catalog_mask, self.accumulative_meta, strict=True
            ):
                if do_crossmatch:
                    result_catalog = result_catalog.crossmatch(**cross_match_kwargs)
                else:
                    result_catalog = result_catalog.map_partitions(
                        skipped_crossmatch, meta_to_match=meta_to_match
                    )

            selected.append(_to_delayed(result_catalog.operation, pixel))

        if len(selected) == 1:
            if self.client is None:
                return _FakeFuture(selected[0].compute())
            return self.client.compute(selected[0])

        combined = dask.delayed(pd.concat)(selected)
        if self.client is None:
            return _FakeFuture(combined.compute())
        return self.client.compute(combined)


class CountMapForPixel:
    """Helper methods to select HEALPix for selective crossmatch"""

    count_maps: list[np.ndarray]

    def __init__(
        self,
        catalog: hats.catalog.Catalog,
        right_catalogs: list[hats.catalog.Catalog],
        *,
        count_fraction: float,
    ):
        """Initialize the CountMapForPixel class.

        Attributes
        ----------
        catalog: hats.catalog.Catalog
            the Anchor catalog
        right_catalogs: list[hats.catalog.Catalog]
            the list of catalogs to crossmatch to
        count_fraction: float
            the fraction of matches above which a pixel is selected for crossmatching
        """
        self.catalog = catalog
        self.right_catalogs = right_catalogs
        self.count_fraction = count_fraction
        self.load_catalog_counts()

    def load_catalog_counts(self):
        """Reads fixed order skymaps from on-disk fits file"""
        count_maps = []
        for cat in [self.catalog, *self.right_catalogs]:
            skymap = np.asarray(read_skymap(cat, None))
            count_maps.append(skymap)
        self.count_maps = count_maps

    def get_pixel_catalog_mask(self, pixel, rng) -> np.ndarray:
        """Get boolean mask for all right catalogs.

        If True, do crossmatch for provided pixel for catalog at index."""
        mask = []
        anchor_counts, right_counts = self.count_maps[0], self.count_maps[1:]
        for counts in right_counts:
            do_crossmatch = get_fraction_at_pixel(pixel, anchor_counts, counts)
            mask.append(do_crossmatch >= self.count_fraction)
        return np.asarray(mask, dtype=bool)


def get_fraction_at_pixel(pixel, counts_a, counts_b) -> float:
    """Gets the expected match fraction for a pixel of the crossmatch of two catalogs."""
    order, ipix = pixel
    order_a = hp.npix2order(len(counts_a))
    order_b = hp.npix2order(len(counts_b))
    if order > order_a or order > order_b:
        raise ValueError("Skymaps order must be as high as the order of the pixel")
    common_order = min(order_a, order_b)
    # TODO: We should probably match the skymap orders only once when we
    # initialize the class, not per partition call, to improve performance
    counts_a = _subpixel_counts(counts_a, order_a, order, common_order, ipix)
    counts_b = _subpixel_counts(counts_b, order_b, order, common_order, ipix)
    total_b = np.sum(counts_b)
    if total_b == 0:
        return 0.0
    n_expected_matches = np.sum(np.minimum(counts_a, counts_b))
    return float(n_expected_matches / total_b)


def _subpixel_counts(counts, order_x, target_order, common_order, ipix) -> np.ndarray:
    """The target pixel's subpixel counts from `counts`, aggregated to `common_order`."""
    f = 4 ** (order_x - target_order)
    seg = counts[ipix * f : (ipix + 1) * f]
    block = 4 ** (order_x - common_order)
    return seg.reshape(-1, block).sum(axis=1)
