import functools

import nested_pandas as npd
import numpy as np
from dask._task_spec import Alias, Task, cull
from dask.dataframe.dask_expr._expr import Expr
from dask.typing import Key
from dask.utils import ensure_dict
from hats import HealpixPixel
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort

from lsdb.operations.functions.divisions import get_pixels_divisions
from lsdb.operations.operation import HealpixGraph, Operation


class FromOperation(Expr):
    """Dask Expression from an LSDB operation."""

    _parameters = ["operation", "_divisions"]

    @functools.cached_property
    def operation_graph(self) -> HealpixGraph:
        """The HealpixGraph built from the operation."""
        return self.operand("operation").build()

    @functools.cached_property
    def sorted_pixels(self) -> list[HealpixPixel]:
        """Returns the list of HEALPix pixels sorted in healpix order."""
        pixels = list(self.operation_graph.pixel_to_key_map.keys())
        sorted_pixels = list(np.array(pixels)[get_pixel_argsort(pixels)])
        return sorted_pixels

    @property
    def _meta(self):
        return self.operand("operation").meta

    def _divisions(self):
        if not self.operand("_divisions"):
            return [None] * (len(self.sorted_pixels) + 1)
        divisions = get_pixels_divisions(self.sorted_pixels)
        return divisions

    def _layer(self) -> dict:
        graph_dict = ensure_dict(self.operation_graph.graph)
        for i, pixel in enumerate(self.sorted_pixels):
            graph_dict[(self._name, i)] = Alias((self._name, i), self.operation_graph.pixel_to_key_map[pixel])
        return graph_dict

    def _task(self, key: Key, index: int) -> Task:
        raise NotImplementedError("FromOperation does not implement _task; use the _layer instead.")


class FromDaskExpression(Operation):
    """LSDB Operation to create a Dask Expression from an LSDB operation."""

    def __init__(self, expr: Expr, healpix_pixels: list[HealpixPixel]) -> None:
        self._expr = expr
        self._healpix_pixels = healpix_pixels

    @property
    def name(self) -> str:
        return f"FromDaskExpression({self._expr})"

    @functools.cached_property
    def key_name(self) -> str:
        return f"from_expr-{self._expr.__dask_tokenize__()}"

    @property
    def meta(self) -> npd.NestedFrame:
        return self._expr._meta  # pylint: disable=protected-access

    @property
    def dependencies(self) -> list[Operation]:
        return []

    @property
    def healpix_pixels(self) -> list[HealpixPixel]:
        return self._healpix_pixels

    def build(self, pixels=None) -> HealpixGraph:
        graph = self._expr.__dask_graph__()
        last_dask_keys = self._expr.__dask_keys__()
        pixel_to_key_map = dict(zip(self._healpix_pixels, last_dask_keys))
        if pixels is not None:
            pixel_to_key_map = {
                pixel: pixel_to_key_map[pixel] for pixel in pixels if pixel in pixel_to_key_map
            }
            graph = cull(graph, list(pixel_to_key_map.values()))
        return HealpixGraph(graph, pixel_to_key_map)
