import hats as hc
import nested_dask as nd

from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.types import DaskDFPixelMap


class AssociationCatalog(HealpixDataset):
    """LSDB Association Catalog DataFrame to perform join analysis of sky catalogs and efficient
    spatial operations.

    Attributes:
        hc_structure: `hats.AssociationCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.AssociationCatalog

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.AssociationCatalog,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure)

    def _create_modified_hc_structure(
        self, hc_structure=None, updated_schema=None, **kwargs
    ) -> hc.catalog.AssociationCatalog:
        """Copy the catalog structure and override the specified catalog info parameters.

        Returns:
            A copy of the catalog's structure with updated info parameters.
        """
        if hc_structure is None:
            hc_structure = self.hc_structure
        return hc_structure.__class__(
            catalog_info=hc_structure.catalog_info.copy_and_update(**kwargs),
            pixels=hc_structure.pixel_tree,
            join_pixels=hc_structure.join_info,
            catalog_path=hc_structure.catalog_path,
            schema=hc_structure.schema if updated_schema is None else updated_schema,
            moc=hc_structure.moc,
        )
