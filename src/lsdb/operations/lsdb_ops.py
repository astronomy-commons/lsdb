from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Self, Callable

from dask._task_spec import Task, TaskRef, cull
from dask.tokenize import _tokenize_deterministic
from dask.utils import funcname

import pandas as pd
import nested_pandas as npd
from hats import HealpixPixel
from pyarrow.lib import Sequence

from lsdb.operations.operation import HealpixGraph, Operation

if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


class FromHealpixMap(Operation):
    def __init__(self, func, pixels, *args, meta=None, map_kwargs=None, **kwargs):
        self.func = func
        self.pixels = pixels
        self.args = args
        self._meta = meta
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
        return f"{funcname(self.func)}-{_tokenize_deterministic(*self.args, self.kwargs)}"

    @property
    def meta(self) -> npd.NestedFrame:
        if self._meta is not None:
            return self._meta
        else:
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
            task = Task(key, self.func, pixel, *self.args, **self.kwargs, **map_kwargs)
            graph[key] = task
            pixel_keys[pixel] = key
        return HealpixGraph(graph, pixel_keys)


class FromSinglePartition(FromHealpixMap):
    def __init__(self, partition, pixel):
        meta = partition.iloc[:0].copy()
        super().__init__(lambda pix, df: df, [pixel], partition, meta=meta)

    @property
    def is_reloadable(self) -> bool:
        return False


class EmptyOperation(FromHealpixMap):
    def __init__(self, meta):
        super().__init__(None, [], meta=meta)

    @property
    def is_reloadable(self) -> bool:
        return False


def map_parts_meta(func, base_meta: npd.NestedFrame, *args, include_pixel=False, **kwargs) -> npd.NestedFrame:
    try:
        if include_pixel:
            result = func(base_meta, HealpixPixel(0, 0), *args, **kwargs)
        else:
            result = func(base_meta, *args, **kwargs)
    except Exception as e:
        raise ValueError(
            "Cannot infer meta for MapPartitions. Either make sure your function works with an"
            " empty dataframe input, or supply a meta for your function"
        ) from e
    return _coerce_to_meta(result)


def _coerce_to_meta(result) -> npd.NestedFrame:
    """Coerce a function result to an empty npd.NestedFrame for use as meta."""
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
        return npd.NestedFrame({"result": pd.Series(dtype=result.dtype)})
    if isinstance(result, dict):
        return npd.NestedFrame(
            {
                k: pd.Series(
                    dtype=pd.api.types.pandas_dtype(type(v[0] if hasattr(v, "__len__") and len(v) > 0 else v))
                )
                for k, v in result.items()
            }
        )
    if isinstance(result, (list, tuple)):
        return npd.NestedFrame(
            {"result": pd.Series(dtype=pd.api.types.pandas_dtype(type(result[0])) if result else object)}
        )
    # scalar
    return npd.NestedFrame({"result": pd.Series(dtype=pd.api.types.pandas_dtype(type(result)))})


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
        f"meta must be a DataFrame, dict of {{name: dtype}}, or list of (name, dtype) tuples, got {type(meta)}"
    )


class MapPartitions(Operation):
    class_func = None
    class_include_pixels = None

    def __init__(self, base: Operation, func, *args, meta=None, include_pixel=False, **kwargs):
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
        self.kwargs = kwargs

    @property
    def func(self) -> Callable:
        if self.class_func is None:
            return self._func
        return self.class_func

    @property
    def name(self) -> str:
        return f"MapPartitions({funcname(self.func)}, {self.base.name})"

    @functools.cached_property
    def key_name(self) -> str:
        return f"{funcname(self.func)}-{_tokenize_deterministic(self.base.meta, self.base.key_name, self.args, self.kwargs)}"

    @property
    def meta(self) -> npd.NestedFrame:
        if self._meta is not None:
            return self._meta
        else:
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
        previous = self.base.build()
        graph = previous.graph
        pixel_keys = {}
        func = self.func
        include_pixel = self.include_pixel

        def wrapped_func(df, _partition_index, *args, **kwargs):
            try:
                if include_pixel:
                    # pixel is already injected as first arg by the include_pixel path
                    result = func(df, *args, **kwargs)
                else:
                    result = func(df, *args, **kwargs)
                return _coerce_to_frame(result)
            except Exception as e:
                if include_pixel and args:
                    raise RuntimeError(
                        f"Error applying function {funcname(func)} to partition {_partition_index}, pixel {args[0]}: {e}"
                    ) from e
                raise RuntimeError(
                    f"Error applying function {funcname(func)} to partition {_partition_index}: {e}"
                ) from e

        for i, (pixel, prev_key) in enumerate(previous.pixel_to_key_map.items()):
            args = self.args
            if self.include_pixel:
                args = (HealpixPixel(*pixel),) + args
            key = (self.key_name, i)
            task = Task(key, wrapped_func, TaskRef(prev_key), i, *args, **self.kwargs)
            graph[key] = task
            pixel_keys[pixel] = key
        return HealpixGraph(graph, pixel_keys)


def perform_select_columns(df, columns):
    return df[columns]


class SelectColumns(MapPartitions):
    @staticmethod
    def class_func(df, item):
        return df[item]

    @property
    def column_selector(self):
        return self.args[0]


class SelectPixels(Operation):
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
        previous = self.base.build()
        selected_pixels = self.pixels
        for p in selected_pixels:
            if p not in previous.pixel_to_key_map:
                raise ValueError(f"Selected Pixel {p} not found in operation")
        selected_keys = [previous.pixel_to_key_map[p] for p in selected_pixels]
        culled_graph = cull(previous.graph, selected_keys)
        pixel_keys = {p: k for p, k in zip(selected_pixels, selected_keys)}
        return HealpixGraph(culled_graph, pixel_keys)


class AlignAndApply(Operation):
    def __init__(
        self,
        input_cats: Sequence[HealpixDataset],
        pixel_lists: Sequence[Sequence[HealpixPixel | None]],
        func,
        meta,
        output_pixels: Sequence[HealpixPixel],
        *args,
        **kwargs,
    ):
        self.input_cats = input_cats
        self.pixel_lists = pixel_lists
        if len(self.input_cats) != len(self.pixel_lists):
            raise ValueError("Inccorect Align and Apply Setup")
        self.func = func
        self._meta = meta
        self.output_pixels = output_pixels
        self.args = args
        self.kwargs = kwargs

    @property
    def input_ops(self) -> list[Operation | None]:
        return [cat._operation if cat is not None else None for cat in self.input_cats]

    @property
    def dependencies(self) -> list[Operation]:
        return [op for op in self.input_ops if op is not None]

    @property
    def metas(self):
        return [op.meta if op is not None else None for op in self.input_ops]

    @property
    def catalog_infos(self):
        return [cat.hc_structure.catalog_info if cat is not None else None for cat in self.input_cats]

    @property
    def name(self) -> str:
        return f"AlignAndApply({funcname(self.func)}, {', '.join(op.name if op is not None else 'None' for op in self.input_ops)})"

    @functools.cached_property
    def key_name(self) -> str:
        key_names = [op.key_name if op is not None else None for op in self.input_ops]
        return f"{funcname(self.func)}-{_tokenize_deterministic(*self.metas, *key_names, *self.pixel_lists, *self.catalog_infos, *self.args, self.kwargs)}"

    @property
    def meta(self) -> npd.NestedFrame:
        return self._meta

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        return list(self.output_pixels)

    def build(self) -> HealpixGraph:
        input_ops = self.input_ops
        graphs = [op.build() if op is not None else None for op in input_ops]
        metas = self.metas
        catalog_infos = self.catalog_infos
        graph = {}
        pixel_key_map = {}
        for g in graphs:
            if g is not None:
                graph = graph | g.graph
        for i, all_pixels in enumerate(zip(self.output_pixels, *self.pixel_lists)):
            output_pixel = all_pixels[0]
            pixels = all_pixels[1:]
            task_refs = []
            for g, m, p in zip(graphs, metas, pixels):
                if g is None:
                    task_refs.append(None)
                elif p is None or p not in g.pixel_to_key_map:
                    task_refs.append(m)
                else:
                    task_refs.append(TaskRef(g.pixel_to_key_map[p]))
            args = task_refs + list(pixels) + catalog_infos + list(self.args)
            kwargs = self.kwargs
            key = (self.key_name, i)
            task = Task(key, self.func, *args, **kwargs)
            graph[key] = task
            pixel_key_map[output_pixel] = key
        culled_graph = cull(graph, list(pixel_key_map.values()))
        return HealpixGraph(culled_graph, pixel_key_map)
