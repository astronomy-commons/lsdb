from __future__ import annotations

from typing import List

import numpy as np
from hipscat.pixel_math import HealpixPixel

from lsdb.core.search.pixel_search import PixelSearch


# pylint: disable=protected-access
class PartitionIndexer:
    """Class that implements the square brackets accessor for catalog partitions."""

    def __init__(self, catalog):
        self.catalog = catalog

    def __getitem__(self, item):
        indices = self._parse_partition_indices(item)
        pixels = self._get_pixels_from_partition_indices(indices)
        return self.catalog._search(PixelSearch(pixels), fine=False)

    def _parse_partition_indices(self, item: int | List[int]) -> List[int]:
        """Parses the partition indices provided in the square brackets accessor.
        It is either a single integer or a sequence-like set of integers."""
        if isinstance(item, int):
            item = [item]
        indices = np.arange(len(self.catalog._ddf_pixel_map), dtype=object)[item].tolist()
        return indices

    def _get_pixels_from_partition_indices(self, indices: List[int]) -> List[HealpixPixel]:
        """Performs a reverse-lookup in the catalog pixel-to-partition map and returns the
        pixels for the specified partition `indices`."""
        inverted_pixel_map = {i: pixel for pixel, i in self.catalog._ddf_pixel_map.items()}
        filtered_pixels = [inverted_pixel_map[key] for key in indices]
        return filtered_pixels
