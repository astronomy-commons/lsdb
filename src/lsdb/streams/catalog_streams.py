from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional, cast

import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.delayed import Delayed, delayed
from dask.distributed import Client, Future
from hats.io.file_io.file_io import read_parquet_dataset
from hats.io.paths import pixel_catalog_file
from hats.io.skymap import read_skymap
from hats.pixel_math.healpix_pixel import HealpixPixel
from hats.pixel_math.spatial_index import compute_spatial_index, spatial_index_to_healpix
from hats.pixel_tree import PixelAlignmentType, align_trees
from upath import UPath

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

        # submitting the next partition request before waiting for the result speeds things up ~10%
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
    """Stream left crossmatches, optionally skipping low-information right catalogs.

    Parameters
    ----------
    catalog : lsdb.Catalog
        Anchor catalog whose rows are preserved.
    *crossmatch_kwargs : dict
        One dictionary per right catalog, containing ``other`` and any arguments
        for ``Catalog.crossmatch``. ``how`` is ``"left"`` (default) or ``"outer"``.
        Each dictionary may additionally specify
        ``compression_columns`` (a nonempty list of science column names) and
        ``compression_ratio_threshold`` (a finite, nonnegative minimum ratio).
        Both must be supplied to enable compression filtering for that catalog.
        Nested columns may be selected as a group or with dotted subcolumn names.
    client : dask.distributed.Client or None, default None
        Client used to compute stream chunks, or None for synchronous execution.
    partitions_per_chunk : int, default 1
        Number of partitions per chunk.
    seed : int or None, default None
        Random seed for partition and row shuffling.
    count_fraction_threshold : float
        Minimum estimated match coverage required to crossmatch a right catalog.

    Notes
    -----
    The compression ratio is the sum of compressed Parquet column-chunk sizes
    divided by the sum of uncompressed sizes across selected columns, row groups,
    and right-catalog source partitions overlapping the current pixel. Higher
    ratios mean less-compressible data; compression overhead can produce ratios
    above one. Finer pixels inherit their source partition's score, which is
    unaffected by row filters or transformations.

    Both coverage and compression thresholds must pass. Skipping preserves anchor
    rows and fills right-catalog columns with nulls. Omitting compression options
    retains coverage-only filtering.

    Scoring reads only footer metadata, with no dedicated cache or data
    computation. It requires on-disk right catalogs and raises an error for
    missing selected columns or unusable metadata.

    Every right catalog is matched against the anchor's rows only (a star around
    the anchor), never against another right catalog. With ``how="outer"``, a right
    catalog's rows that match no anchor row are also emitted, but only from the sky
    its crossmatch already reads: the anchor's pixels, kept where the anchor has data
    (``outer_footprint="left"`` in ``Catalog.crossmatch``). Each such row has its own
    ``_ra``/``_dec`` and null anchor and other right-catalog columns. Anchor rows carry
    the anchor's coordinates in ``_ra``/``_dec``. Outer right catalogs need margin
    caches on the anchor and the right catalog (see ``Catalog.crossmatch``). An object
    detected by two right catalogs but not by the anchor is emitted as two separate rows.

    Examples
    --------
    Configure compression filtering independently for each right catalog:

    >>> from lsdb.streams.catalog_streams import CrossMatchStream
    >>> stream = CrossMatchStream(  # doctest: +SKIP
    ...     anchor,
    ...     {"other": photometry, "compression_columns": ["flux_g", "flux_r"],
    ...      "compression_ratio_threshold": 0.5},
    ...     {"other": spectra},  # Coverage filtering only.
    ...     count_fraction_threshold=0.1,
    ... )
    """

    def __init__(
        self,
        catalog: Catalog,
        *crossmatch_kwargs: dict[str, Any],
        client: Client | None = None,
        partitions_per_chunk: int = 1,
        seed: int | None = None,
        count_fraction_threshold: float,
    ) -> None:
        super().__init__(
            catalog,
            client=client,
            partitions_per_chunk=partitions_per_chunk,
            seed=seed,
        )

        all_kwargs = []
        self.compression_filters: list[tuple[list[str], float] | None] = []
        self.right_how: list[str] = []
        for kwargs in crossmatch_kwargs:
            new_kwargs = kwargs.copy()
            how = new_kwargs.pop("how", "left")
            if how not in ("left", "outer"):
                raise ValueError("CrossMatchStream supports how='left' or how='outer' for each right catalog")
            self.right_how.append(how)
            columns = new_kwargs.pop("compression_columns", None)
            threshold = new_kwargs.pop("compression_ratio_threshold", None)
            if columns is None and threshold is None:
                self.compression_filters.append(None)
            else:
                configuration_error = (
                    "Specify both a nonempty list of compression_columns and a finite, "
                    "nonnegative compression_ratio_threshold."
                )
                if (
                    not isinstance(columns, list)
                    or not columns
                    or not all(isinstance(column, str) and column for column in columns)
                ):
                    raise ValueError(configuration_error)
                if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold < 0:
                    raise ValueError(configuration_error)
                if kwargs["other"].hc_structure.catalog_base_dir is None:
                    raise ValueError("Compression filtering requires an on-disk right catalog.")
                self.compression_filters.append((columns.copy(), float(threshold)))
            new_kwargs["suffixes"] = ("", "_" + kwargs["other"].name)
            new_kwargs["suffix_method"] = "all_columns"
            new_kwargs["how"] = "left"
            all_kwargs.append(new_kwargs)
        self.crossmatch_kwargs = all_kwargs

        self.accumulative_meta = []
        result_catalog = self.catalog
        for right_catalog_kwargs in self.crossmatch_kwargs:
            result_catalog = result_catalog.crossmatch(**right_catalog_kwargs).map_partitions(
                lambda df: df.drop(columns=["_dist_arcsec"])
            )
            self.accumulative_meta.append(result_catalog.meta)

        # Stream the crossmatch's aligned pixels, which can outnumber the anchor's partitions
        self._pixels = result_catalog._operation.healpix_pixels
        self.partitions_per_chunk = min(partitions_per_chunk, self.npartitions)

        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng((1 << 32, self.seed))

        right_catalogs = [kwargs["other"] for kwargs in self.crossmatch_kwargs]
        self.mask_generator = CountMapForPixel(
            self.catalog, right_catalogs, count_fraction=count_fraction_threshold
        )

        # With only left joins the chain above is already a star: each crossmatch
        # matches against the anchor's coordinates. Outer joins need the star built
        # explicitly, so that right-only rows are never matched to the next catalog.
        self._star = self._plan_star() if "outer" in self.right_how else None
        if self._star is not None:
            self._pixels = self._star.pixels
            self.partitions_per_chunk = min(partitions_per_chunk, self.npartitions)

    def _plan_star(self) -> "_StarPlan":
        """Crossmatch the anchor with each right catalog separately."""
        anchor = self.catalog.map_partitions(_number_anchor_rows)
        results = []
        right_columns = []
        for kwargs, how in zip(self.crossmatch_kwargs, self.right_how, strict=True):
            star_kwargs = {**kwargs, "how": how}
            if how == "outer":
                star_kwargs["outer_footprint"] = "left"
            results.append(anchor.crossmatch(**star_kwargs))
            suffix = kwargs["suffixes"][1]
            right_columns.append([f"{column}{suffix}" for column in kwargs["other"].columns])

        # Draw the finest pixels of all the crossmatches together. Each crossmatch keeps
        # only the pixels near anchor data, at its own resolution, and emits its unmatched
        # right rows anywhere in those pixels, so every one of its pixels must be drawn.
        pixel_tree = results[0].hc_structure.pixel_tree
        for result in results[1:]:
            pixel_tree = align_trees(
                pixel_tree, result.hc_structure.pixel_tree, alignment_type=PixelAlignmentType.OUTER
            ).pixel_tree

        outer_meta = next(result.meta for result in results if "_ra" in result.meta.columns)
        meta = pd.concat(
            [self.catalog.meta]
            + [result.meta[columns] for result, columns in zip(results, right_columns, strict=True)]
            + [outer_meta[["_ra", "_dec"]]],
            axis=1,
        )
        # The outer crossmatches above already required the anchor's coordinate columns
        catalog_info = self.catalog.hc_structure.catalog_info
        assembly = _StarAssembly(
            ra_column=cast(str, catalog_info.ra_column),
            dec_column=cast(str, catalog_info.dec_column),
            right_columns=right_columns,
            outer=[how == "outer" for how in self.right_how],
            meta=npd.NestedFrame(meta),
        )
        return _StarPlan(
            anchor=anchor, results=results, pixels=pixel_tree.get_healpix_pixels(), assembly=assembly
        )

    def _right_catalog_mask(self, pixel: HealpixPixel) -> list[bool]:
        """Whether to crossmatch each right catalog at ``pixel``: coverage and compression must pass."""
        mask = []
        for kwargs, do_crossmatch, compression_filter in zip(
            self.crossmatch_kwargs,
            self.mask_generator.get_pixel_catalog_mask(pixel),
            self.compression_filters,
            strict=True,
        ):
            if do_crossmatch and compression_filter is not None:
                columns, threshold = compression_filter
                do_crossmatch = _get_compression_ratio(kwargs["other"], pixel, columns) >= threshold
            mask.append(bool(do_crossmatch))
        return mask

    def _compute(self, selected: list[Delayed], empty: npd.NestedFrame) -> Future | _FakeFuture:
        """Compute the selected partitions, concatenated, with the client if there is one."""
        if len(selected) == 0:
            return _FakeFuture(empty)

        if len(selected) == 1:
            if self.client is None:
                return _FakeFuture(selected[0].compute())
            return cast(Future, self.client.compute(selected[0]))

        combined = delayed(pd.concat)(selected)
        if self.client is None:
            return _FakeFuture(combined.compute())
        return cast(Future, self.client.compute(combined))

    def _submit_star_partitions(self, partitions: np.ndarray) -> Future | _FakeFuture:
        """Submit star crossmatches of the anchor at the given pixels."""
        star = cast(_StarPlan, self._star)
        selected = []
        for pixel_index in partitions:
            pixel = self._pixels[pixel_index]
            right_parts = [
                _partition_delayeds(result, pixel) if do_crossmatch else None
                for result, do_crossmatch in zip(star.results, self._right_catalog_mask(pixel), strict=True)
            ]
            anchor_parts = _partition_delayeds(star.anchor, pixel)
            selected.append(delayed(_assemble_star_pixel)(pixel, anchor_parts, right_parts, star.assembly))
        return self._compute(selected, star.assembly.meta)

    def submit_next_partitions(self, partitions: np.ndarray) -> Future | _FakeFuture:
        """Submit the next set of partitions for computation."""
        if self._star is not None:
            return self._submit_star_partitions(partitions)

        # The crossmatch may leave no pixels at all, when the pixel does not overlap
        # with one of the right catalogs, so we use whatever keys the build has
        # instead of looking the left pixel up.
        def _to_delayed(operation):
            build = operation.build()
            return [Delayed(key, build.graph) for key in build.keys]

        selected = []

        for pixel_index in partitions:
            pixel = self._pixels[pixel_index]

            def skipped_crossmatch(
                partition: npd.NestedFrame, *, meta_to_match: npd.NestedFrame
            ) -> npd.NestedFrame:
                old_n_columns = partition.shape[1]
                new_columns = meta_to_match.columns[old_n_columns:]
                for column in new_columns:
                    partition[column] = pd.Series(index=partition.index, dtype=meta_to_match.dtypes[column])
                return partition

            result_catalog = self.catalog.search(PixelSearch(pixel, fine=True))
            for cross_match_kwargs, do_crossmatch, meta_to_match in zip(
                self.crossmatch_kwargs,
                self._right_catalog_mask(pixel),
                self.accumulative_meta,
                strict=True,
            ):
                if do_crossmatch:
                    result_catalog = result_catalog.crossmatch(**cross_match_kwargs).map_partitions(
                        lambda df: df.drop(columns=["_dist_arcsec"])
                    )
                else:
                    result_catalog = result_catalog.map_partitions(
                        skipped_crossmatch, meta_to_match=meta_to_match
                    )

            selected.extend(_to_delayed(result_catalog._operation))  # pylint: disable=protected-access

        return self._compute(selected, self.accumulative_meta[-1])


_ANCHOR_ROW = "_lsdb_anchor_row"


def _number_anchor_rows(partition: npd.NestedFrame) -> npd.NestedFrame:
    """Number each anchor partition's rows, so star crossmatch results merge on them."""
    numbers = pd.Series(np.arange(len(partition)), index=partition.index, dtype=pd.ArrowDtype(pa.int64()))
    return partition.assign(**{_ANCHOR_ROW: numbers})


@dataclass
class _StarAssembly:
    """What a star crossmatch task needs to assemble one pixel's rows."""

    ra_column: str
    dec_column: str
    right_columns: list[list[str]]
    outer: list[bool]
    meta: npd.NestedFrame


@dataclass
class _StarPlan:
    """The anchor, with numbered rows, crossmatched with each right catalog separately."""

    anchor: Catalog
    results: list[Catalog]
    pixels: list[HealpixPixel]
    assembly: _StarAssembly


def _partition_delayeds(catalog: Catalog, pixel: HealpixPixel) -> list[Delayed]:
    """Delayed partitions of ``catalog`` that overlap ``pixel``."""
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
    """``block`` with exactly the columns of ``meta``, filling missing ones with nulls."""
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
    right_parts: list[list[npd.NestedFrame] | None],
    assembly: _StarAssembly,
) -> npd.NestedFrame:
    """Merge each right catalog's matches onto the anchor rows in ``pixel``.

    ``right_parts`` holds, per right catalog, the partitions of its crossmatch with
    the anchor that cover ``pixel``, or None when it is skipped there. Outer right
    catalogs also contribute their unmatched rows inside ``pixel``.
    """
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
    rows["_ra"] = rows[assembly.ra_column]
    rows["_dec"] = rows[assembly.dec_column]
    blocks = [_conform(block, assembly.meta) for block in [rows, *unmatched]]
    return npd.NestedFrame(pd.concat(blocks))


def _get_compression_ratio(catalog: Catalog, pixel: HealpixPixel, columns: list[str]) -> float:
    """Read selected-column footer sizes from overlapping source partitions."""
    structure = catalog.hc_structure
    source_pixels = PixelSearch(pixel).filter_hc_catalog(structure).get_healpix_pixels()
    if not source_pixels:
        return 0.0
    path_generator = (
        pixel_catalog_file
        if catalog.loading_config is None
        else catalog.loading_config.path_generator or pixel_catalog_file
    )

    compressed_bytes = 0
    uncompressed_bytes = 0
    for source_pixel in source_pixels:
        path = path_generator(
            cast(UPath, structure.catalog_base_dir), source_pixel, None, structure.catalog_info.npix_suffix
        )
        # Dataset fragments also support HATS pixels stored as directories. Reuse
        # their metadata (including any footer already read during discovery).
        _, dataset = read_parquet_dataset(path)
        for fragment in dataset.get_fragments():
            metadata = fragment.metadata
            matched_columns = set()
            for row_group_index in range(metadata.num_row_groups):
                row_group = metadata.row_group(row_group_index)
                for column_index in range(row_group.num_columns):
                    chunk = row_group.column(column_index)
                    name = chunk.path_in_schema.replace(".list.element", "")
                    matches = [
                        column for column in columns if name == column or name.startswith(column + ".")
                    ]
                    if not matches:
                        continue
                    matched_columns.update(matches)
                    if chunk.total_compressed_size < 0 or chunk.total_uncompressed_size <= 0:
                        raise ValueError(f"Unusable compression metadata for {name!r} in {fragment.path}.")
                    compressed_bytes += chunk.total_compressed_size
                    uncompressed_bytes += chunk.total_uncompressed_size
            if set(columns) - matched_columns:
                raise ValueError(
                    f"Compression columns {sorted(set(columns) - matched_columns)} "
                    f"are missing from footer metadata in {fragment.path}."
                )
    if uncompressed_bytes == 0:
        raise ValueError(f"No usable compression metadata for pixel {pixel} in catalog {catalog.name!r}.")
    return compressed_bytes / uncompressed_bytes


class CountMapForPixel:
    """Helper methods to select HEALPix for selective crossmatch"""

    def __init__(
        self,
        catalog: Catalog,
        right_catalogs: list[Catalog],
        *,
        count_fraction: float,
    ):
        """Initialize the CountMapForPixel class.

        Attributes
        ----------
        catalog: lsdb.Catalog
            the Anchor catalog
        right_catalogs: list[lsdb.Catalog]
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
        left_count_map = np.asarray(read_skymap(self.catalog.hc_structure, None))
        right_count_maps = []
        for cat in self.right_catalogs:
            skymap = np.asarray(read_skymap(cat.hc_structure, None))
            right_count_maps.append(skymap)
        self.left_count_map = left_count_map
        self.right_count_maps = right_count_maps
        self.minimums = [
            self.reorder_skymap_and_get_min(left_count_map, right_count_map)
            for right_count_map in right_count_maps
        ]

    def reorder_skymap_and_get_min(self, counts_a, counts_b):
        """Reorder two skymaps to a common order and return the minimum counts."""
        order_a = hp.npix2order(len(counts_a))
        order_b = hp.npix2order(len(counts_b))
        common_order = min(order_a, order_b)

        # Reorder counts_a to common_order
        if order_a > common_order:
            counts_a = counts_a.reshape(-1, 4 ** (order_a - common_order)).sum(axis=1)

        # Reorder counts_b to common_order
        if order_b > common_order:
            counts_b = counts_b.reshape(-1, 4 ** (order_b - common_order)).sum(axis=1)

        # Return the minimum of the two skymaps
        return np.minimum(counts_a, counts_b)

    def get_pixel_catalog_mask(self, pixel) -> np.ndarray:
        """Get boolean mask for all right catalogs.

        If True, do crossmatch for provided pixel for catalog at index."""
        mask = []
        for right_counts, pair_minimums in zip(self.right_count_maps, self.minimums):
            do_crossmatch = get_fraction_at_pixel(pixel, pair_minimums, right_counts)
            mask.append(do_crossmatch >= self.count_fraction)
        return np.asarray(mask, dtype=bool)


def get_fraction_at_pixel(pixel, minimums, right_counts) -> float:
    """Gets the expected match fraction for a pixel of the crossmatch of two catalogs."""
    total_minimums = get_sum_at_pixel(pixel, minimums)
    total_right_counts = get_sum_at_pixel(pixel, right_counts)
    if total_right_counts == 0:
        return 0.0
    return float(total_minimums / total_right_counts)


def get_sum_at_pixel(pixel, counts) -> int:
    """Gets the sum of counts for a pixel of a catalog."""
    order, ipix = pixel
    order_counts = hp.npix2order(len(counts))
    selected_counts = counts[ipix * 4 ** (order_counts - order) : (ipix + 1) * 4 ** (order_counts - order)]
    return selected_counts.sum()
