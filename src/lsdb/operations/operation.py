from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import nested_pandas as npd


class HealpixGraph:
    """Task Graph where each node corresponds to a HEALPix pixel"""

    def __init__(self, graph: dict, pixel_to_key_map: dict):
        self.graph = graph
        self.pixel_to_key_map = pixel_to_key_map

    @property
    def keys(self):
        return list(self.pixel_to_key_map.values())


class Operation(ABC):
    allow_column_projection_passthrough = False

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the Task"""
        pass

    @property
    @abstractmethod
    def key_name(self) -> str:
        pass

    @abstractmethod
    def build(self) -> HealpixGraph:
        """Returns the task graph for this operation, where each node corresponds to a HEALPix pixel"""
        pass

    @property
    @abstractmethod
    def meta(self) -> npd.NestedFrame:
        """Returns the metadata for the output of this operation"""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> list[Operation]:
        """Returns the list of input operations for this operation"""
        pass

    @property
    @abstractmethod
    def healpix_pixels(self) -> list[HealpixGraph]:
        """Returns the list of HEALPix pixels that this operation's partitions correspond to"""
        pass

    def optimize(self) -> Self:
        return self

    def __repr__(self):
        return self.name
