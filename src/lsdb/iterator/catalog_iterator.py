from collections.abc import Iterator
from typing import Optional, Self

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

    def result(self) -> pd.DataFrame:
        """Return the held value."""
        return self.obj


class CatalogIterator(Iterator[pd.DataFrame]):
    """Iterator yielding random subsets of partitions from a catalog.

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

    Examples
    --------
    Consider a toy catalog, which contains 12 data partitions:

    >>> import lsdb
    >>> cat = lsdb.generate_catalog(500, 10, seed=1)
    >>> cat.npartitions
    12

    A simple example of iterating through a catalog in chunks of 4 (random) partitions at a time:

    >>> cat_iter = lsdb.CatalogIterator(catalog=cat, partitions_per_chunk=4, seed=1)
    >>> for chunk in cat_iter:
    ...     print(len(chunk))
    156
    186
    158

    Alternatively, you can loop through the data multiple times by setting `iter_limit`:

    >>> cat_iter = lsdb.CatalogIterator(catalog=cat, partitions_per_chunk=4, iter_limit=3, seed=1)
    >>> for chunk in cat_iter:
    ...     print(len(chunk))
    173
    159
    183
    168
    158
    154
    173
    154
    178

    Finally, you can loop through the data infinitely by setting `iter_limit=None`:

    >>> cat_iter = lsdb.CatalogIterator(catalog=cat, partitions_per_chunk=4, iter_limit=None, seed=1)
    >>> for _ in range(5):
    ...     chunk = next(cat_iter)
    ...     print(len(chunk))
    161
    185
    189
    159
    158
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

        self.future: Optional[Future | _FakeFuture] = self._submit_next_partitions()

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
        futurable = sliced_catalog._ddf  # pylint: disable=protected-access

        if self.client is None:
            future = _FakeFuture(dask.compute(futurable)[0])
        else:
            future = self.client.compute(futurable)
        return future

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pd.DataFrame:
        # check the future as well, for typing reasons
        # but it shouldn't be None if the iterator is not empty.
        if self._empty or self.future is None:
            raise StopIteration("All partitions have been processed")

        result: pd.DataFrame = self.future.result()
        result = result.sample(frac=1, random_state=self.rng)

        if len(self.partitions_left) > 0:
            self.future = self._submit_next_partitions()
        else:
            self._empty = True
            self.future = None

        return result

    def __len__(self) -> int:
        return int(np.ceil(len(self.partitions_left) / self.partitions_per_chunk))
