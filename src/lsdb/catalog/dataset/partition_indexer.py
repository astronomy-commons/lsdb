from typing import Sequence

import numpy as np

from lsdb.core.search.pixel_search import PixelSearch


class PartitionIndexer:
    """Class that implements the slicing accessor for catalog partitions."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        indices = self._parse_indices(item)
        pixels = self._get_pixels_from_partition_indices(indices)
        search = PixelSearch(pixels)
        return self.dataset._search(search, fine=False)

    def _parse_indices(self, item):
        if isinstance(item, int):
            item = [item]
        # pylint: disable=protected-access
        index = np.arange(len(self.dataset._ddf_pixel_map), dtype=object)[item].tolist()
        return index

    def _get_pixels_from_partition_indices(self, indices: Sequence[int]):
        # pylint: disable=protected-access
        pixel_map = self.dataset._ddf_pixel_map
        inverted_pixel_map = {i: pixel for pixel, i in pixel_map.items()}
        filtered_pixels = [inverted_pixel_map[key] for key in indices]
        return filtered_pixels
