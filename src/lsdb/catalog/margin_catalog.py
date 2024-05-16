import dask.dataframe as dd
import healpy as hp
import hipscat as hc
import numpy as np
from hipscat.pixel_tree.moc_filter import filter_by_moc
from mocpy import MOC

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
                f"Cannot Filter Margin: Margin size {self.hc_structure.catalog_info.margin_threshold} is "
                f"greater than the size of a pixel at the highest order {max_order}."
            )

        margin_search_moc = metadata.pixel_tree.to_moc().degrade_to_order(max_order).add_neighbours()

        filtered_hc_structure = self.hc_structure.filter_by_moc(margin_search_moc)
        ddf_partition_map, search_ddf = self._perform_search(
            metadata, filtered_hc_structure.get_healpix_pixels(), search, fine
        )
        return self.__class__(search_ddf, ddf_partition_map, filtered_hc_structure)
