from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import nested_pandas as npd
from hats import HealpixPixel


class HealpixGraph:
    """Task Graph where each node corresponds to a HEALPix pixel"""

    def __init__(self, graph: dict, pixel_to_key_map: dict):
        self.graph = graph
        self.pixel_to_key_map = pixel_to_key_map

    @property
    def keys(self):
        """Returns the list of keys in the graph."""
        return list(self.pixel_to_key_map.values())


class Operation(ABC):
    """Abstract base class defining an operation."""

    allow_column_projection_passthrough = False
    preserves_pixel_stats = False

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        """The name of the Task"""
        pass

    @property
    @abstractmethod
    def key_name(self) -> str:  # pragma: no cover
        """The name of the key in the task graph that corresponds to this operation's output"""
        pass

    @abstractmethod
    def build(self, pixels: list[HealpixPixel] | None = None) -> HealpixGraph:  # pragma: no cover
        """Returns the task graph for this operation, where each node corresponds to a HEALPix pixel"""
        pass

    @property
    @abstractmethod
    def meta(self) -> npd.NestedFrame:  # pragma: no cover
        """Returns the metadata for the output of this operation"""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> list[Operation]:  # pragma: no cover
        """Returns the list of input operations for this operation"""
        pass

    @property
    def is_reloadable(self) -> bool:  # pragma: no cover
        """Whether this operation can be reconstructed from disk via _reload_with_filter."""
        return False

    @property
    @abstractmethod
    def healpix_pixels(self) -> list[HealpixPixel]:  # pragma: no cover
        """Returns the list of HEALPix pixels that this operation's partitions correspond to"""
        pass

    def optimize(self) -> Self:  # pragma: no cover
        """Returns an optimized version of the operation."""
        return self

    def __repr__(self):  # pragma: no cover
        """String representation of the operation."""
        return self.name

    @property
    def pixel_stats_preserved(self) -> bool:
        """Whether this operation and all its dependencies preserve the on-disk statistics."""
        return self.preserves_pixel_stats and all(dep.pixel_stats_preserved for dep in self.dependencies)
