from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional, cast

import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.spatial_index import compute_spatial_index, spatial_index_to_healpix
from hats.pixel_tree.pixel_alignment import align_trees
from hats.pixel_tree.pixel_alignment_types import PixelAlignmentType
from mocpy import MOC

from lsdb import Catalog, PixelSearch


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

    The following grabs 4 random partitions 3 times in a row, looping over the data as needed:

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

    @property
    def npartitions(self) -> int:
        """Number of partitions the stream draws from."""
        return len(self._pixels)

    def get_next_partitions(
        self, partitions_left: np.ndarray, rng: np.random.Generator  # pylint: disable=unused-argument
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the next set of partitions to iterate over."""
        # Chomp a subset of partitions when running once through the data
        return (
            partitions_left[: -self.partitions_per_chunk],
            partitions_left[-self.partitions_per_chunk :],
        )

    def _delayed_partition(self, pixel: HealpixPixel) -> Delayed:
        build = self.operation.build(pixels=[pixel])
        return Delayed(build.pixel_to_key_map[pixel], build.graph)

    def submit_next_partitions(self, partitions: np.ndarray) -> Future | _FakeFuture:
        """Submit the next set of partitions for computation."""
        selected = [self._delayed_partition(self._pixels[i]) for i in partitions]

        if len(selected) == 1:
            if self.client is None:
                return _FakeFuture(selected[0].compute())
            return cast(Future, self.client.compute(selected[0]))

        combined = delayed(pd.concat)(selected)
        if self.client is None:
            return _FakeFuture(combined.compute())
        return cast(Future, self.client.compute(combined))

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
    168
    162
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
            return self.rng.permutation(self.iterable.npartitions)
        return np.arange(self.iterable.npartitions)

    def _get_next_partitions(self) -> np.ndarray:
        """Get the next set of partitions to process."""
        self.partitions_left, partitions = self.iterable.get_next_partitions(self.partitions_left, self.rng)
        return partitions

    def __iter__(self) -> "CatalogIterator":
        return self

    def __next__(self) -> pd.DataFrame:
        if self._empty or self.future is None:
            raise StopIteration("All partitions have been processed")

        if len(self.partitions_left) > 0:
            next_future = self.iterable.submit_next_partitions(self._get_next_partitions())
        else:
            self._empty = True
            next_future = None

        result: pd.DataFrame = self.future.result()

        if self.iterable.shuffle:
            result = result.sample(frac=1, random_state=self.rng)

        self.future = next_future
        return result

    def __len__(self) -> int:
        # Fail gracefully if the iterable is an InfiniteStream
        if isinstance(self.iterable, InfiniteStream):
            raise TypeError("Length is not defined for an InfiniteStream.")

        return int(np.ceil(len(self.partitions_left) / self.iterable.partitions_per_chunk))


class CrossMatchStream(InfiniteStream):
    """Stream an anchor catalog crossmatched with each of several right catalogs.

    Each right catalog is left- or outer-joined to the anchor.

    Parameters
    ----------
    catalog : lsdb.Catalog
        Anchor catalog whose rows are preserved.
    *crossmatch_kwargs : dict
        One dictionary per right catalog, containing ``other`` and any arguments
        for ``Catalog.crossmatch``. ``how`` is ``"left"`` (default) or ``"outer"``.
        Anchor columns keep their names; ``suffixes`` sets only the right catalog's
        column suffix (default ``"_" + other.name``), which must make every output
        column name unique.
    client : dask.distributed.Client or None, default None
        Client used to compute stream chunks, or None for synchronous execution.
    partitions_per_chunk : int, default 1
        Number of partitions per chunk.
    seed : int or None, default None
        Random seed for partition and row shuffling.
    footprint : mocpy.MOC or None, default None
        Sky region the stream covers, clipped to the anchor's pixel tree. None uses
        the anchor's data coverage (``anchor.hc_structure.moc``, or its pixel tree if
        it has no MOC), so every anchor row is kept and unmatched right rows come only
        from sky where the anchor has data. A given MOC also restricts the anchor; pass
        ``anchor.hc_structure.pixel_tree.to_moc()`` to emit unmatched right rows
        anywhere in the anchor's pixels.
    Notes
    -----
    Every right catalog is crossmatched with the anchor separately (a star around
    the anchor), never with another right catalog, and its matches are merged onto
    the anchor rows. With ``how="outer"``, a right catalog's rows that match no anchor
    row are also emitted, but only inside ``footprint``. Each such row has null
    anchor and other right-catalog columns.
    With any outer right catalog the stream adds ``_ra``/``_dec`` columns: the
    anchor's coordinates for anchor rows, and the right catalog's own for unmatched
    right rows. Outer right catalogs need margin caches on the anchor and the right
    catalog (see ``Catalog.crossmatch``). An object detected by two right catalogs but
    not by the anchor is emitted as two separate rows.

    Examples
    --------
    Left-join photometry and outer-join spectra to an anchor catalog:

    >>> from lsdb.streams.catalog_streams import CrossMatchStream
    >>> stream = CrossMatchStream(  # doctest: +SKIP
    ...     anchor,
    ...     {"other": photometry},
    ...     {"other": spectra, "how": "outer"},
    ...     partitions_per_chunk=4,
    ... )
    """

    def __init__(
        self,
        catalog: Catalog,
        *crossmatch_kwargs: dict[str, Any],
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        seed: int | None = None,
        footprint: MOC | None = None,
    ) -> None:
        catalog, self.footprint = _restrict_to_footprint(catalog, footprint)
        super().__init__(
            catalog,
            client=client,
            partitions_per_chunk=partitions_per_chunk,
            seed=seed,
        )

        all_kwargs = []
        for kwargs in crossmatch_kwargs:
            new_kwargs = kwargs.copy()
            if new_kwargs.setdefault("how", "left") not in ("left", "outer"):
                raise ValueError("CrossMatchStream supports how='left' or how='outer' for each right catalog")
            # Anchor columns keep their names, so only the right suffix is the caller's to set
            new_kwargs["suffixes"] = ("", (kwargs.get("suffixes") or ("", "_" + kwargs["other"].name))[1])
            new_kwargs["suffix_method"] = "all_columns"
            all_kwargs.append(new_kwargs)
        self.crossmatch_kwargs = all_kwargs

        # Stream the crossmatches' aligned pixels, which can outnumber the anchor's partitions
        self._anchor, self._results, self._pixels, self._assembly = self._plan_star()
        self.partitions_per_chunk = min(partitions_per_chunk, self.npartitions)

    def _plan_star(self) -> tuple[Catalog, list[Catalog], list[HealpixPixel], "_StarAssembly"]:
        anchor = self.catalog.map_partitions(_number_anchor_rows)
        results = []
        right_columns = []
        outer = []
        for kwargs in self.crossmatch_kwargs:
            result = anchor.crossmatch(**kwargs)
            outer.append(kwargs["how"] == "outer")
            if outer[-1]:
                # Rows are placed by _ra/_dec, so this keeps unmatched right rows inside the
                # footprint and drops right-only pixels before they are read
                result = result.moc_search(self.footprint)
            results.append(result)
            suffix = kwargs["suffixes"][1]
            right_columns.append([f"{column}{suffix}" for column in kwargs["other"].columns])

        # Draw the finest pixels of all the crossmatches together. Each crossmatch keeps
        # every anchor pixel with data, split at its own resolution, and an outer one emits
        # its unmatched right rows anywhere in its pixels, so all of them must be drawn.
        trees = [result.hc_structure.pixel_tree for result in results] or [anchor.hc_structure.pixel_tree]
        pixel_tree = trees[0]
        for tree in trees[1:]:
            pixel_tree = align_trees(pixel_tree, tree, alignment_type=PixelAlignmentType.OUTER).pixel_tree

        # Outer crossmatches add _ra/_dec, so that unmatched right rows have coordinates
        coordinates = [result.meta[["_ra", "_dec"]] for result, is_outer in zip(results, outer) if is_outer]
        meta = pd.concat(
            [self.catalog.meta]
            + [result.meta[columns] for result, columns in zip(results, right_columns, strict=True)]
            + coordinates[:1],
            axis=1,
        )
        if meta.columns.duplicated().any():
            raise ValueError(
                f"Output columns {sorted(set(meta.columns[meta.columns.duplicated()]))} are repeated: "
                "give each right catalog distinct suffixes=('', '_<label>')."
            )
        # The crossmatches above already required the anchor's coordinate columns
        catalog_info = self.catalog.hc_structure.catalog_info
        assembly = _StarAssembly(
            ra_column=cast(str, catalog_info.ra_column),
            dec_column=cast(str, catalog_info.dec_column),
            right_columns=right_columns,
            outer=outer,
            meta=npd.NestedFrame(meta),
        )
        return anchor, results, pixel_tree.get_healpix_pixels(), assembly

    def _delayed_partition(self, pixel: HealpixPixel) -> Delayed:
        right_parts = [_partition_delayeds(result, pixel) for result in self._results]
        anchor_parts = _partition_delayeds(self._anchor, pixel)
        return delayed(_assemble_star_pixel)(pixel, anchor_parts, right_parts, self._assembly)


def _data_coverage(catalog: Catalog) -> MOC:
    structure = catalog.hc_structure
    pixel_coverage = structure.pixel_tree.to_moc()
    return pixel_coverage if structure.moc is None else structure.moc.intersection(pixel_coverage)


def _restrict_to_footprint(catalog: Catalog, footprint: MOC | None) -> tuple[Catalog, MOC]:
    """Clip ``footprint`` to the anchor's pixel tree, so every drawn pixel has anchor rows.

    The default, the anchor's data coverage, needs no anchor filtering: its rows lie in its MOC."""
    if footprint is None:
        return catalog, _data_coverage(catalog)
    footprint = footprint.intersection(catalog.hc_structure.pixel_tree.to_moc())
    return catalog.moc_search(footprint), footprint


_ANCHOR_ROW = "_lsdb_anchor_row"


def _number_anchor_rows(partition: npd.NestedFrame) -> npd.NestedFrame:
    """Number each anchor partition's rows, so star crossmatch results merge on them."""
    numbers = pd.Series(np.arange(len(partition)), index=partition.index, dtype=pd.ArrowDtype(pa.int64()))
    return partition.assign(**{_ANCHOR_ROW: numbers})


@dataclass
class _StarAssembly:
    ra_column: str
    dec_column: str
    right_columns: list[list[str]]
    outer: list[bool]
    meta: npd.NestedFrame


def _partition_delayeds(catalog: Catalog, pixel: HealpixPixel) -> list[Delayed]:
    pixels = PixelSearch(pixel).filter_hc_catalog(catalog.hc_structure).get_healpix_pixels()
    build = catalog._operation.build(pixels=pixels)  # pylint: disable=protected-access
    return [Delayed(build.pixel_to_key_map[partition_pixel], build.graph) for partition_pixel in pixels]


def _rows_in_pixel(frame: npd.NestedFrame, pixel: HealpixPixel, ra_column: str, dec_column: str):
    """Rows of ``frame`` whose coordinates fall in ``pixel``, as HATS partitions them."""
    if len(frame) == 0:
        return frame
    spatial_index = compute_spatial_index(
        frame[ra_column].to_numpy(dtype=float), frame[dec_column].to_numpy(dtype=float)
    )
    return frame[spatial_index_to_healpix(spatial_index, target_order=pixel.order) == pixel.pixel]


def _conform(block: pd.DataFrame, meta: npd.NestedFrame) -> npd.NestedFrame:
    columns = {
        column: (
            block[column]
            if column in block.columns
            else pd.Series([None] * len(block), index=block.index, dtype=meta.dtypes[column])
        )
        for column in meta.columns
    }
    return npd.NestedFrame(columns, index=block.index)


def _assemble_star_pixel(
    pixel: HealpixPixel,
    anchor_parts: list[npd.NestedFrame],
    right_parts: list[list[npd.NestedFrame]],
    assembly: _StarAssembly,
) -> npd.NestedFrame:
    """Merge each right catalog's matches onto the anchor rows in ``pixel``, and append
    outer catalogs' unmatched rows."""
    rows = _rows_in_pixel(pd.concat(anchor_parts), pixel, assembly.ra_column, assembly.dec_column).copy()
    unmatched = []
    for parts, columns, outer in zip(right_parts, assembly.right_columns, assembly.outer, strict=True):
        if not parts:
            continue
        result = pd.concat(parts)
        anchored = result[_ANCHOR_ROW].notna().to_numpy()
        matches = result.loc[anchored, [_ANCHOR_ROW, *columns]].set_index(_ANCHOR_ROW)
        rows = rows.join(matches, on=_ANCHOR_ROW)
        if outer:
            unmatched.append(_rows_in_pixel(result.loc[~anchored], pixel, "_ra", "_dec"))
    if any(assembly.outer):
        rows["_ra"] = rows[assembly.ra_column]
        rows["_dec"] = rows[assembly.dec_column]
    blocks = [_conform(block, assembly.meta) for block in [rows, *unmatched]]
    return npd.NestedFrame(pd.concat(blocks))
