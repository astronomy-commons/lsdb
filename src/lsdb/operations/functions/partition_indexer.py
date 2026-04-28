from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from hats import HealpixPixel

from lsdb.core.search.region_search import PixelSearch

if TYPE_CHECKING:
    from lsdb import HealpixDataset


# pylint: disable=protected-access
class PartitionIndexer:
    """Class that implements the square brackets accessor for catalog partitions."""

    def __init__(self, cat: HealpixDataset):
        self.cat = cat

    def __getitem__(self, item):
        if isinstance(item, (int, HealpixPixel)):
            item = [item]
        if all(isinstance(i, HealpixPixel) for i in item):
            pixels = np.array(item)
        else:
            pixels = np.array(self.cat.get_healpix_pixels())[item]
        return self.cat.search(PixelSearch(pixels))

    def __iter__(self) -> Iterator:
        for pixel in self.cat.get_healpix_pixels():
            yield self.cat.search(PixelSearch([pixel]))
