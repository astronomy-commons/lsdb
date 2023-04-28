from typing import Dict, Tuple

import dask.dataframe as dd
import hipscat as hc
from hipscat.pixel_math import HealpixPixel, HealpixInputTypes, get_healpix_pixel

from lsdb.catalog.catalog import DaskDFPixelMap
from lsdb.catalog.dataset.dataset import Dataset


AssociationPixelMap = Dict[Tuple[HealpixPixel, HealpixPixel], int]


class AssociationCatalog(Dataset):

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
            self,
            ddf: dd.core.DataFrame,
            ddf_pixel_map: AssociationPixelMap,
            hc_structure: hc.catalog.AssociationCatalog
    ):
        super().__init__(ddf, hc_structure)
        self.ddf_pixel_map = ddf_pixel_map

    def get_partition_index(self, primary_pixel: HealpixInputTypes, join_pixel: HealpixInputTypes):
        primary_pixel = get_healpix_pixel(primary_pixel)
        join_pixel = get_healpix_pixel(join_pixel)
        return self.ddf_pixel_map[(primary_pixel, join_pixel)]
