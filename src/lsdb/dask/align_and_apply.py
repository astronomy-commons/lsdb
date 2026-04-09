from __future__ import annotations

import functools
from functools import partial
from typing import Callable, TYPE_CHECKING

import pandas as pd
import nested_pandas as npd
from dask._collections import new_collection
from dask._task_spec import Task, TaskRef, Dict
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.dask_expr._expr import (
    Blockwise,
    Projection,
    determine_column_projection,
    Expr,
    _DelayedExpr,
)
from dask.delayed import Delayed
from dask.tokenize import _tokenize_deterministic
from dask.typing import Key
from dask.utils import funcname
from hats import HealpixPixel


if TYPE_CHECKING:
    from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def make_align_and_apply_expr(
    catalog_mappings: list[tuple[HealpixDataset | None, list[HealpixPixel]]],
    func: Callable,
    meta: npd.NestedFrame | pd.DataFrame | pd.Series,
    *args,
    enforce_metadata: bool = True,
    divisions=None,
    parent_meta=None,
    required_columns=None,
    **kwargs,
):
    catalogs, pixel_lists = zip(*catalog_mappings)
    catalogs = list(catalogs)
    pixel_lists = list(pixel_lists)

    args = [_DelayedExpr(a) if isinstance(a, Delayed) else a for a in args]
    newkwargs = {}
    delayed_kwargs = []
    for k, v in kwargs.items():
        if isinstance(v, Delayed):
            dexpr = _DelayedExpr(v)
            delayed_kwargs.append(dexpr)
            newkwargs[k] = TaskRef(dexpr.__dask_keys__()[0])
        else:
            newkwargs[k] = v
    del kwargs
    new_expr = AlignAndApply(
        catalogs,
        pixel_lists,
        func,
        meta,
        enforce_metadata,
        divisions,
        parent_meta,
        required_columns,
        newkwargs.pop("token", None),
        Dict(newkwargs),
        len(args),
        *args,
        *delayed_kwargs,
    )
    return new_collection(new_expr)


def _get_catalog_arg(catalog: HealpixDataset | None, pixel: HealpixPixel | None):
    if catalog is None:
        return None
    if pixel is None or pixel not in catalog.hc_structure.pixel_tree:
        return catalog._ddf._meta.copy()
    return TaskRef((catalog._ddf.expr._name, catalog.get_partition_index(pixel.order, pixel.pixel)))


class AlignAndApply(Blockwise):
    _parameters = [
        "catalogs",
        "pixel_lists",
        "func",
        "meta",
        "enforce_metadata",
        "user_divisions",
        "parent_meta",
        "required_columns",
        "token",
        "kwargs",
        "nargs",
    ]
    _defaults: dict = {
        "kwargs": None,
        "user_divisions": None,
        "parent_meta": None,
        "required_columns": None,
        "token": None,
        "nargs": 0,
    }

    @functools.cached_property
    def token(self):
        if "token" in self._parameters:
            return self.operand("token")
        return None

    def __str__(self):
        return f"AlignAndApply({funcname(self.func)})"

    @property
    def deterministic_token(self):
        if not self._determ_token:
            # Just tokenize self to fall back on __dask_tokenize__
            # Note how this differs to the implementation of __dask_tokenize__
            operands = [
                [cat._ddf.expr if cat is not None else None for cat in self.catalogs]
            ] + self.operands[1:]
            self._determ_token = _tokenize_deterministic(type(self), operands)
        return self._determ_token

    @functools.cached_property
    def _name(self):
        if self.token is not None:
            head = self.token
        else:
            head = funcname(self.func).lower()
        return f"{head}-{self.deterministic_token}"

    @functools.cached_property
    def _num_partitions(self):
        return len(self.pixel_lists[0])

    def _broadcast_dep(self, dep: Expr):
        # Always broadcast single-partition dependencies in MapPartitions
        return dep.npartitions == 1

    @functools.cached_property
    def args(self):
        return self.operands[len(self._parameters) : len(self._parameters) + self.nargs]

    @functools.cached_property
    def _meta(self):
        meta = self.operand("meta")
        if meta is None:
            raise ValueError("Meta must be provided for AlignAndApply operations")
        return meta

    def dependencies(self):
        deps = super().dependencies()
        for cat in self.catalogs:
            if cat is not None:
                deps.append(cat._ddf.expr)
        return deps

    def _divisions(self):
        divisions = self.operand("user_divisions")

        if divisions is None:
            return (None,) * (self._num_partitions + 1)

        if len(divisions) != self._num_partitions + 1:
            raise ValueError(f"Length of divisions must be {self._num_partitions + 1}, got {len(divisions)}")

        return divisions

    def _task(self, name: Key, index: int) -> Task:
        normal_args = [self._blockwise_arg(op, index) for op in self.args]
        pixels = [pixel_list[index] for pixel_list in self.pixel_lists]
        catalog_args = [_get_catalog_arg(cat, pixel) for cat, pixel in zip(self.catalogs, pixels)]
        catalog_infos = [cat.hc_structure.catalog_info if cat is not None else None for cat in self.catalogs]
        args = catalog_args + pixels + catalog_infos + normal_args
        kwargs = dict(self.kwargs if self.kwargs is not None else {})

        if self.enforce_metadata:
            kwargs.update(
                {
                    "_func": self.func,
                    "_meta": self._meta,
                }
            )
            return Task(name, apply_and_enforce, *args, **kwargs)
        else:
            return Task(
                name,
                self.func,
                *args,
                **kwargs,
            )

    @staticmethod
    def projected_operation(mapped_func, post_projection, *args, **kwargs):
        # Apply a mapped function and then project columns.
        # Used by `_simplify_up` to apply column projection.
        return mapped_func(*args, **kwargs)[post_projection]

    def _simplify_up(self, parent, dependents):
        if isinstance(parent, Projection) and self.required_columns is not None:
            if missing := set(self.required_columns) - set(self.frame.columns):
                raise KeyError(f"Some elements of `required_columns` are missing: {missing}")

            columns = determine_column_projection(
                self, parent, dependents, additional_columns=self.required_columns
            )
            columns = [col for col in self.frame.columns if col in columns]

            if columns == self.frame.columns:
                # Don't add unnecessary Projections
                return

            return type(parent)(
                type(self)(
                    self.frame[columns],
                    partial(self.projected_operation, self.func, parent.columns),
                    self.meta[parent.columns],
                    *self.operands[3:],
                ),
                *parent.operands[1:],
            )
