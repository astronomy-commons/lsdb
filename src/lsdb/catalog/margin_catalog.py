import hats as hc
import nested_dask as nd
import pyarrow as pa
from hats.io import paths

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.core.search.abstract_search import AbstractSearch
from lsdb.types import DaskDFPixelMap


class MarginCatalog(HealpixDataset):
    """LSDB Catalog DataFrame to contain the "margin" of another HATS catalog.
    spatial operations.

    Attributes:
        hc_structure: `hats.MarginCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.MarginCatalog

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.MarginCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)

    def search(self, search: AbstractSearch):
        """Find rows by reusable search algorithm.

        Filters partitions in the catalog to those that match some rough criteria and their neighbors.
        Filters to points that match some finer criteria.

        Args:
            search (AbstractSearch): Instance of AbstractSearch.

        Returns:
            A new Catalog containing the points filtered to those matching the search parameters.
        """
        filtered_hc_structure = search.filter_hc_catalog(self.hc_structure)
        ddf_partition_map, search_ndf = self._perform_search(filtered_hc_structure, search)
        return self.__class__(search_ndf, ddf_partition_map, filtered_hc_structure)


def _validate_margin_catalog(margin_hc_catalog, hc_catalog):
    """Validate that the margin and main catalogs have compatible schemas"""
    full_schema = _create_margin_schema(hc_catalog.schema)
    if not full_schema.equals(margin_hc_catalog.schema):
        raise ValueError("The margin catalog and the main catalog must have the same schema")


def _create_margin_schema(main_catalog_schema: pa.Schema) -> pa.Schema:
    """Create a pyarrow schema for the margin catalog from the main catalog schema."""
    order_field = pa.field(f"margin_{paths.PARTITION_ORDER}", pa.uint8())
    dir_field = pa.field(f"margin_{paths.PARTITION_DIR}", pa.uint64())
    pixel_field = pa.field(f"margin_{paths.PARTITION_PIXEL}", pa.uint64())
    return main_catalog_schema.append(order_field).append(dir_field).append(pixel_field)
