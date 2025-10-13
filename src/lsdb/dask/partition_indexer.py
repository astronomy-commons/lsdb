from __future__ import annotations

import numpy as np

from lsdb.core.search.region_search import PixelSearch


# pylint: disable=protected-access
class PartitionIndexer:
    """Class that implements the square brackets accessor for catalog partitions."""

    def __init__(self, catalog):
        self.catalog = catalog

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        indices = np.arange(len(self.catalog._ddf_pixel_map), dtype=np.int64)[item].tolist()
        pixels = self._get_pixels_from_partition_indices(indices)
        return self.catalog.search(PixelSearch(pixels))

    def _get_pixels_from_partition_indices(self, indices: list[int]) -> list[tuple[int, int]]:
        """Performs a reverse-lookup in the catalog pixel-to-partition map and returns the
        pixels for the specified partition `indices`."""
        inverted_pixel_map = {i: pixel for pixel, i in self.catalog._ddf_pixel_map.items()}
        filtered_pixels = [inverted_pixel_map[key] for key in indices]
        return [(p.order, p.pixel) for p in filtered_pixels]
