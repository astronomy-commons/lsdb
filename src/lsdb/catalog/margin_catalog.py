import dask.dataframe as dd
import hipscat as hc
import numpy as np
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.filter import get_filtered_pixel_list
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap


class MarginCatalog(HealpixDataset):
    """LSDB Catalog DataFrame to contain the "margin" of another HiPSCat catalog.
    spatial operations.

    Attributes:
        hc_structure: `hipscat.MarginCatalog` object representing the structure
                      and metadata of the HiPSCat catalog
    """

    hc_structure: hc.catalog.MarginCatalog

    def __init__(
        self,
        ddf: dd.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.MarginCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)

    def _search(self, search):
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria.
        Filters to points that match some finer criteria.

        Args:
            search: instance of AbstractSearch

        Returns:
            A new Catalog containing the points filtered to those matching the search parameters.
        """
        filtered_search_pixels = search.search_partitions(self.hc_structure.get_healpix_pixels())
        margin_order = max(pixel.order for pixel in filtered_search_pixels) + 1
        margin_pixels = [hc.pixel_math.get_margin(pixel.order, pixel.pixel, margin_order - pixel.order) for pixel in filtered_search_pixels]
        margin_pixels = list(set(np.concatenate(margin_pixels)))
        margin_pixels = [HealpixPixel(margin_order, pixel) for pixel in margin_pixels]
        margin_pixel_tree = PixelTreeBuilder.from_healpix(margin_pixels)
        filtered_margin_pixels = get_filtered_pixel_list(self.hc_structure.pixel_tree, margin_pixel_tree)
        filtered_pixels = list(set(filtered_search_pixels + filtered_margin_pixels))
        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(filtered_pixels)
        ddf_partition_map, search_ddf = self._perform_search(filtered_pixels, search)
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)
