from collections.abc import Iterator
from typing import Self

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, Future

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

    def result(self):
        return self.obj


class CatalogIterator(Iterator[pd.Series | pd.DataFrame]):
    """Generator yielding training data from an LSDB

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
    iter_limit : int or None, default 1
        The number of times to loop through the data. By default, the iterator
        will loop through the data once. If set to a value greater than 1, the
        iterator will loop through the data that many times. The repeated
        partitions are shuffled together, so the same partition may be seen
        multiple times in a row, or not at all until the end of the first pass.
        If None, loops infinitely over the data, yielding a different random
        subset of partitions each time. Changing this value affects the
        randomness for a given seed.
    seed : int
        Random seed to use for observation sampling.

    Methods
    -------
    __next__() -> pd.Series
        Provides light curves as a nested series.
    """

    def __init__(
        self,
        *,
        catalog: Catalog,
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        iter_limit: bool | int = 1,
        seed: int | None = None,
    ) -> None:
        self.catalog = catalog
        self.client = client
        self.partitions_per_chunk = min(partitions_per_chunk, self.catalog.npartitions)
        self.iter_limit = iter_limit
        self.seed = seed

        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng((1 << 32, self.seed))

        if self.iter_limit is not None and self.iter_limit > 1:
            repeated_partitions = np.repeat(np.arange(self.catalog.npartitions), iter_limit)
            self.partitions_left = self.rng.permutation(repeated_partitions)

        else:
            self.partitions_left = self.rng.permutation(self.catalog.npartitions)
        self._empty = False

        self.future = self._submit_next_partitions()

    def _get_next_partitions(self) -> np.ndarray:
        # Get a random subset of partitions when looping infinitely
        if self.iter_limit is None:
            return self.rng.choice(self.partitions_left, self.partitions_per_chunk, replace=False)

        # Chomp a subset of partitions when running once through the data
        self.partitions_left, partitions = (
            self.partitions_left[: -self.partitions_per_chunk],
            self.partitions_left[-self.partitions_per_chunk :],
        )
        return partitions

    def _submit_next_partitions(self) -> Future | _FakeFuture:
        partitions = self._get_next_partitions()
        sliced_catalog = self.catalog.partitions[partitions]

        futurable = sliced_catalog._ddf if hasattr(sliced_catalog, "_ddf") else sliced_catalog

        if self.client is None:
            future = _FakeFuture(dask.compute(futurable)[0])
        else:
            future = self.client.compute(futurable)
        return future

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pd.Series:
        if self._empty:
            raise StopIteration("All partitions have been processed")

        result: pd.Series | pd.DataFrame = self.future.result()
        result = result.sample(frac=1, random_state=self.rng)

        if len(self.partitions_left) > 0:
            self.future = self._submit_next_partitions()
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        return int(np.ceil(len(self.partitions_left) / self.partitions_per_chunk))
