from collections.abc import Iterator
from typing import Optional

import dask
import numpy as np
import pandas as pd
from dask.delayed import Delayed
from dask.distributed import Client, Future
from dask.optimization import cull

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

        self.client = client
        self.partitions_per_chunk = min(partitions_per_chunk, self.catalog.npartitions)
        self.shuffle = shuffle
        self.seed = seed

        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng((1 << 32, self.seed))

        # Pre-extract delayed partitions with one-time graph optimization.
        # This avoids repeated graph construction, optimization, and catalog
        # search/reload overhead on every iteration.
        #
        # Crucially, to_delayed() returns Delayed objects that all share the
        # same full graph. We cull each to its own subgraph so that
        # client.compute() only sends the minimal per-partition graph to the
        # scheduler, avoiding O(N^2) graph transmission overhead.
        raw_delayed = catalog.to_delayed(optimize_graph=True)
        shared_graph = dict(raw_delayed[0].__dask_graph__())
        self._delayed_partitions = []
        for d in raw_delayed:
            culled_graph, _ = cull(shared_graph, list(d.__dask_keys__()))
            self._delayed_partitions.append(Delayed(d.key, culled_graph))

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
        selected = [self._delayed_partitions[i] for i in partitions]

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
