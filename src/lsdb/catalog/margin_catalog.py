import hats as hc
import nested_dask as nd
from hats.io import paths

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
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


def _validate_margin_catalog(margin_hc_catalog, hc_catalog):
    """Validate that the margin and main catalogs have compatible schemas. The order of
    the pyarrow fields should not matter."""
    margin_catalog_fields = set((f.name, f.type) for f in margin_hc_catalog.schema)
    main_catalog_fields = set((f.name, f.type) for f in hc_catalog.schema)

    dropped_fields = main_catalog_fields - margin_catalog_fields
    dropped_fields = [f for f in dropped_fields if f[0] not in paths.HIVE_COLUMNS]

    added_fields = margin_catalog_fields - main_catalog_fields
    added_fields = [f for f in added_fields if f[0] not in paths.HIVE_COLUMNS]

    if len(dropped_fields) or len(added_fields):
        raise ValueError("The margin catalog and the main catalog must have the same schema.")
