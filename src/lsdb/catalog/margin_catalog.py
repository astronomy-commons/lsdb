import dask.dataframe as dd
import healpy as hp
import hipscat as hc
import numpy as np
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.filter import get_filtered_pixel_list
from hipscat.pixel_tree.pixel_tree_builder import PixelTreeBuilder

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.core.search.abstract_search import AbstractSearch
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
        ddf: dd.core.DataFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.MarginCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)

    def _search(self, metadata: hc.catalog.Catalog, search: AbstractSearch, fine: bool = True):
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria and their neighbors.
        Filters to points that match some finer criteria.

        Args:
            search (AbstractSearch): Instance of AbstractSearch.
            fine (bool): True if points are to be filtered, False if not. Defaults to True.

        Returns:
            A new Catalog containing the points filtered to those matching the search parameters.
        """

        # if the margin size is greater than the size of a pixel, this is an invalid search
        max_order = self.hc_structure.pixel_tree.get_max_depth()
        max_order_size = hp.nside2resol(2**max_order, arcmin=True)
        if self.hc_structure.catalog_info.margin_threshold > max_order_size * 60:
            raise ValueError(
                f"Margin size {self.hc_structure.catalog_info.margin_threshold} is greater than the size of "
                f"a pixel at the highest order {max_order}."
            )

        # Get the pixels that match the search pixels
        filtered_search_pixels = search.search_partitions(self.hc_structure.get_healpix_pixels())

        # Get the margin pixels at the max order + 1 from the search pixels
        # the get_margin function requires a higher order than the given pixel
        margin_order = max(pixel.order for pixel in filtered_search_pixels) + 1
        margin_pixels = [
            hc.pixel_math.get_margin(pixel.order, pixel.pixel, margin_order - pixel.order)
            for pixel in filtered_search_pixels
        ]

        # Remove duplicate margin pixels and construct HealpixPixel objects
        margin_pixels = list(set(np.concatenate(margin_pixels)))
        margin_pixels = [HealpixPixel(margin_order, pixel) for pixel in margin_pixels]

        # Align the margin pixels with the catalog pixels and combine with the search pixels
        margin_pixel_tree = PixelTreeBuilder.from_healpix(margin_pixels)
        filtered_margin_pixels = get_filtered_pixel_list(self.hc_structure.pixel_tree, margin_pixel_tree)
        filtered_pixels = list(set(filtered_search_pixels + filtered_margin_pixels))

        filtered_hc_structure = self.hc_structure.filter_from_pixel_list(filtered_pixels)
        ddf_partition_map, search_ddf = self._perform_search(metadata, filtered_pixels, search, fine)
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)
