from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Sequence

import nested_pandas as npd
import numpy as np
import pandas as pd
from dask._task_spec import Task, TaskRef, cull
from dask.dataframe.utils import check_meta
from dask.tokenize import _tokenize_deterministic
from dask.utils import funcname
from hats import HealpixPixel

from lsdb.operations.operation import HealpixGraph, Operation

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def run_and_verify_meta(func, meta, *args, **kwargs):
    """Run func and verify that its output matches the provided meta."""
    result = func(*args, **kwargs)
    if not isinstance(result, meta.__class__):
        raise ValueError(
            f"Function returned result of type {type(result)}, but meta is of type {type(meta)}. "
            "Function must return the same type as meta for meta inference to work correctly."
        )
    check_meta(result, meta, funcname=funcname(func))
    return result


def _verified(func, meta):
    """Wrap func with run_and_verify_meta while preserving func's name for dask."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_and_verify_meta(func, meta, *args, **kwargs)

    return wrapper


class FromHealpixMap(Operation):
    """An Operation that constructs a HealpixGraph from a function that maps HealpixPixels to DataFrames."""

    def __init__(self, func, pixels, *args, meta=None, map_kwargs=None, verify_meta=True, **kwargs):
        self.func = func
        self.pixels = pixels
        self.args = args
        self._meta = meta
        self.verify_meta = verify_meta
        if map_kwargs is not None:
            for k in map_kwargs:
                if k in kwargs:
                    raise ValueError(f"Cannot specify {k} in both map_kwargs and kwargs for FromHealpixMap")
                if len(map_kwargs[k]) != len(pixels):
                    raise ValueError(
                        f"Length of map_kwargs for {k} must match number of pixels in FromHealpixMap"
                    )
        self.map_kwargs = map_kwargs
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return f"FromHealpixMap({funcname(self.func)})"

    @functools.cached_property
    def key_name(self) -> str:
        tokenized = _tokenize_deterministic(self.func, *self.args, self.kwargs, self.map_kwargs)
        return f"{funcname(self.func)}-{tokenized}"

    @property
    def meta(self) -> npd.NestedFrame | pd.DataFrame:
        if self._meta is not None:
            return self._meta
        first_part = self.func(self.pixels[0], *self.args, **self.kwargs)
        if not isinstance(first_part, pd.DataFrame):
            raise ValueError("FromMap function must return a pandas DataFrame")
        return first_part.iloc[:0].copy()

    @property
    def dependencies(self) -> list[Operation]:
        return []

    @property
    def is_reloadable(self) -> bool:
        return True

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        return self.pixels

    def build(self) -> HealpixGraph:
        graph = {}
        pixel_keys = {}
        for i, pixel in enumerate(self.pixels):
            key = (self.key_name, i)
            map_kwargs = {k: v[i] for k, v in self.map_kwargs.items()} if self.map_kwargs is not None else {}
            func = _verified(self.func, self.meta) if self.verify_meta else self.func
            task = Task(key, func, pixel, *self.args, **self.kwargs, **map_kwargs)
            graph[key] = task
            pixel_keys[pixel] = key
        return HealpixGraph(graph, pixel_keys)


class FromSinglePartition(FromHealpixMap):
    """A FromHealpixMap that constructs a HealpixGraph from a single partition, ignoring the pixel input."""

    def __init__(self, partition, pixel):
        meta = partition.iloc[:0].copy()
        super().__init__(lambda pix, df: df, [pixel], partition, meta=meta)

    @property
    def is_reloadable(self) -> bool:
        return False


class EmptyOperation(FromHealpixMap):
    """An Operation that produces an empty partition with the specified meta."""

    def __init__(self, meta):
        super().__init__(None, [], meta=meta)

    @property
    def is_reloadable(self) -> bool:
        return False


def map_parts_meta(func, base_meta: npd.NestedFrame, *args, include_pixel=False, **kwargs) -> npd.NestedFrame:
    """Infer meta for a MapPartitions operation by running func on an empty DataFrame."""
    try:
        if include_pixel:
            result = func(base_meta.copy(), HealpixPixel(0, 0), *args, **kwargs)
        else:
            result = func(base_meta.copy(), *args, **kwargs)
    except (KeyError, TypeError, ValueError):
        raise  # let meaningful validation errors through unchanged
    except Exception as e:
        raise ValueError(
            "Cannot infer meta for MapPartitions. Either make sure your function works with an"
            " empty dataframe input, or supply a meta for your function"
        ) from e
    return _coerce_to_meta(result)


def _coerce_to_meta(result) -> npd.NestedFrame:
    """Coerce a function result to an empty npd.NestedFrame for use as meta."""

    def _safe_dtype(t: type) -> np.dtype:
        """Return the pandas dtype, falling back to object for unsupported types."""
        try:
            return pd.api.types.pandas_dtype(t)
        except TypeError:
            return np.dtype("object")

    if result is None:
        raise ValueError(
            "Cannot infer meta for MapPartitions. Function returned None for an empty "
            "DataFrame input. Either make sure your function works with an empty DataFrame "
            "input, or supply a meta for your function"
        )
    if isinstance(result, npd.NestedFrame):
        return result.iloc[:0]
    if isinstance(result, pd.DataFrame):
        return npd.NestedFrame(result.iloc[:0])
    if isinstance(result, pd.Series):
        return npd.NestedFrame({"result": result.iloc[:0]})
    if isinstance(result, dict):
        return npd.NestedFrame(
            {
                k: pd.Series(dtype=_safe_dtype(type(v[0] if hasattr(v, "__len__") and len(v) > 0 else v)))
                for k, v in result.items()
            }
        )
    if isinstance(result, (list, tuple)):
        return npd.NestedFrame(
            {"result": pd.Series(dtype=_safe_dtype(type(result[0])) if result else object)}
        )
    # scalar
    return npd.NestedFrame({"result": pd.Series(dtype=_safe_dtype(type(result)))})


def _coerce_to_frame(result) -> npd.NestedFrame:
    """Coerce a partition function result to an npd.NestedFrame at execution time."""
    if isinstance(result, npd.NestedFrame):
        return result
    if isinstance(result, pd.DataFrame):
        return npd.NestedFrame(result)
    if isinstance(result, pd.Series):
        return npd.NestedFrame({"result": result.values}, index=result.index)
    if isinstance(result, dict):
        return npd.NestedFrame({k: [v] if not hasattr(v, "__len__") else v for k, v in result.items()})
    if isinstance(result, (list, tuple)):
        return npd.NestedFrame({"result": result})
    # scalar
    return npd.NestedFrame({"result": [result]})


def _normalize_meta(meta) -> npd.NestedFrame:
    """Normalize meta input to an npd.NestedFrame, accepting the same formats as Dask."""
    if isinstance(meta, npd.NestedFrame):
        return meta
    if isinstance(meta, pd.DataFrame):
        return npd.NestedFrame(meta)
    if isinstance(meta, dict):
        return npd.NestedFrame({k: pd.Series(dtype=v) for k, v in meta.items()})
    if isinstance(meta, (list, tuple)) and all(isinstance(m, tuple) and len(m) == 2 for m in meta):
        # list of (name, dtype) tuples — another Dask-accepted format
        return npd.NestedFrame({k: pd.Series(dtype=v) for k, v in meta})
    raise ValueError(
        f"meta must be a DataFrame, dict of {{name: dtype}}, or list of (name, dtype) tuples, got {type(meta)}"  # pylint: disable=line-too-long
    )


class MapPartitions(Operation):
    """An Operation that applies a function to each partition of a HealpixGraph."""

    class_func = None
    """Subclasses with a predictable function supply this, and a custom user-function cannot be applied in that case"""
    class_include_pixels: bool | None = None
    """Subclasses that determine their own pixels will supply this, and a custom user-defined set of pixels cannot be supplied in that case."""

    def __init__(
        self, base: Operation, func, *args, meta=None, include_pixel=False, verify_meta=True, **kwargs
    ):
        self.base = base
        if self.class_func is not None and func is not None:
            raise ValueError("Cannot specify func for MapPartitions when class_func is set")
        if self.class_include_pixels is not None and include_pixel != self.class_include_pixels:
            raise ValueError(
                "Cannot specify include_pixel for MapPartitions when class_include_pixels is set"
            )
        self._func = func
        self.args = args
        # Ensure that input meta is normalized to a NestedFrame
        self._meta = _normalize_meta(meta) if meta is not None else None
        self.include_pixel = include_pixel
        self.verify_meta = verify_meta
        self.kwargs = kwargs

    @property
    def func(self) -> Callable:
        """Return the function to apply to each partition, using class_func if set."""
        if self.class_func is None:
            return self._func
        return self.class_func

    @property
    def name(self) -> str:
        return f"MapPartitions({funcname(self.func)}, {self.base.name})"

    @functools.cached_property
    def key_name(self) -> str:
        tokenized = _tokenize_deterministic(
            self.func, self.base.meta, self.base.key_name, self.args, self.kwargs
        )
        return f"{funcname(self.func)}-{tokenized}"

    @functools.cached_property
    def meta(self) -> npd.NestedFrame:
        if self._meta is not None:
            return self._meta
        return map_parts_meta(
            self.func, self.base.meta, *self.args, include_pixel=self.include_pixel, **self.kwargs
        )

    @property
    def dependencies(self) -> list[Operation]:
        return [self.base]

    @property
    def is_reloadable(self) -> bool:
        return self.base.is_reloadable

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        return self.base.healpix_pixels

    def build(self) -> HealpixGraph:
        """Build the HealpixGraph from the Operation"""
        previous = self.base.build()
        graph = previous.graph
        pixel_keys = {}
        func = self.func
        include_pixel = self.include_pixel
        meta = self.meta

        def wrapped_func(df, _partition_index, *args, **kwargs):
            try:
                result = func(df, *args, **kwargs)
                return _coerce_to_frame(result)
            except Exception as e:
                if include_pixel and args:
                    raise RuntimeError(
                        f"Error applying function {funcname(func)} to partition {_partition_index}, pixel {args[0]}: {e}"  # pylint: disable=line-too-long
                    ) from e
                raise RuntimeError(
                    f"Error applying function {funcname(func)} to partition {_partition_index}: {e}"
                ) from e

        for i, (pixel, prev_key) in enumerate(previous.pixel_to_key_map.items()):
            args = self.args
            if self.include_pixel:
                args = (HealpixPixel(*pixel),) + args
            key = (self.key_name, i)
            task_func = _verified(wrapped_func, meta) if self.verify_meta else wrapped_func
            task = Task(key, task_func, TaskRef(prev_key), i, *args, **self.kwargs)
            graph[key] = task
            pixel_keys[pixel] = key
        return HealpixGraph(graph, pixel_keys)


def perform_select_columns(df, columns):
    """Helper function to select columns from a DataFrame for SelectColumns operation."""
    return df[columns]


class SelectColumns(MapPartitions):
    """An Operation that selects a subset of columns from each partition of a HealpixGraph."""

    @staticmethod
    def class_func(df, item):
        """Select the specified columns from the DataFrame."""
        return df[item]

    @property
    def column_selector(self):
        """Return the column selector, which is the first argument."""
        return self.args[0]


class SelectPixels(Operation):
    """An Operation that selects a subset of HealpixPixels from a HealpixGraph."""

    def __init__(self, base: Operation, pixels: Sequence[HealpixPixel]):
        self.base = base
        self.pixels = pixels

    @property
    def name(self) -> str:
        return f"SelectPixels({self.base.name})"

    @functools.cached_property
    def key_name(self) -> str:
        return f"select_pixels-{_tokenize_deterministic(self.base.meta, self.base.key_name, *self.pixels)}"

    @property
    def meta(self) -> npd.NestedFrame:
        return self.base.meta

    @property
    def dependencies(self) -> list[Operation]:
        return [self.base]

    @property
    def is_reloadable(self) -> bool:
        return self.base.is_reloadable

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        return list(self.pixels)

    def build(self) -> HealpixGraph:
        """Build the HealpixGraph from the Operation."""
        previous = self.base.build()
        selected_pixels = self.pixels
        for p in selected_pixels:
            if p not in previous.pixel_to_key_map:
                raise ValueError(f"Selected Pixel {p} not found in operation")
        selected_keys = [previous.pixel_to_key_map[p] for p in selected_pixels]
        culled_graph = cull(previous.graph, selected_keys)
        pixel_keys = dict(zip(selected_pixels, selected_keys))
        # pixel_keys = {p: k for p, k in zip(selected_pixels, selected_keys)}
        return HealpixGraph(culled_graph, pixel_keys)


class AlignAndApply(Operation):
    """An Operation that applies a function to aligned partitions from multiple HealpixGraphs."""

    def __init__(
        self,
        input_cats: Sequence[HealpixDataset | None],
        pixel_lists: Sequence[Sequence[HealpixPixel | None]],
        func,
        meta,
        output_pixels: Sequence[HealpixPixel],
        *args,
        verify_meta=True,
        **kwargs,
    ):
        self.input_cats = input_cats
        self.pixel_lists = pixel_lists
        if len(self.input_cats) != len(self.pixel_lists):
            raise ValueError("Incorrect Align and Apply Setup")
        self.func = func
        self._meta = meta
        self.output_pixels = output_pixels
        self.args = args
        self.verify_meta = verify_meta
        self.kwargs = kwargs

    @property
    def input_ops(self) -> list[Operation | None]:
        """Get the operations corresponding to the input catalogs"""
        return [
            cat._operation if cat is not None else None  # pylint: disable=protected-access
            for cat in self.input_cats
        ]

    @property
    def dependencies(self) -> list[Operation]:
        """Get the dependencies of the input operations"""
        return [op for op in self.input_ops if op is not None]

    @property
    def metas(self):
        """Get the metas of the input operations"""
        return [op.meta if op is not None else None for op in self.input_ops]

    @property
    def catalog_infos(self):
        """Get the catalog infos of the input catalogs"""
        return [cat.hc_structure.catalog_info if cat is not None else None for cat in self.input_cats]

    @property
    def name(self) -> str:
        """Return the name of the resulting AlignAndApply operation"""
        op_names = ", ".join(op.name if op is not None else "None" for op in self.input_ops)
        return f"AlignAndApply({funcname(self.func)}, {op_names})"

    @functools.cached_property
    def key_name(self) -> str:
        """Return key names for the resulting operation"""
        key_names = [op.key_name if op is not None else None for op in self.input_ops]
        tokenized = _tokenize_deterministic(
            self.func,
            *self.metas,
            *key_names,
            *self.pixel_lists,
            *self.catalog_infos,
            *self.args,
            self.kwargs,
        )
        return f"{funcname(self.func)}-{tokenized}"

    @property
    def meta(self) -> npd.NestedFrame:
        """Return the dask-style meta for the operation"""
        return self._meta

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        """Return the HealpixPixels corresponding to the output pixels of the operation"""
        return list(self.output_pixels)

    def build(self) -> HealpixGraph:
        """Build the HealpixGraph from the Operation."""
        input_ops = self.input_ops
        graphs = [op.build() if op is not None else None for op in input_ops]
        func = _verified(self.func, self.meta) if self.verify_meta else self.func
        graph: dict = {}
        pixel_key_map = {}
        for g in graphs:
            if g is not None:
                graph = graph | g.graph
        for i, all_pixels in enumerate(zip(self.output_pixels, *self.pixel_lists)):
            output_pixel = all_pixels[0]
            pixels = all_pixels[1:]
            task_refs: list = []
            for g, m, p in zip(graphs, self.metas, pixels):
                if g is None:
                    task_refs.append(None)
                elif p is None or p not in g.pixel_to_key_map:
                    task_refs.append(m)
                else:
                    task_refs.append(TaskRef(g.pixel_to_key_map[p]))
            args = task_refs + list(pixels) + self.catalog_infos + list(self.args)
            kwargs = self.kwargs
            key = (self.key_name, i)
            task = Task(key, func, *args, **kwargs)
            graph[key] = task
            pixel_key_map[output_pixel] = key
        culled_graph = cull(graph, list(pixel_key_map.values()))
        return HealpixGraph(culled_graph, pixel_key_map)
