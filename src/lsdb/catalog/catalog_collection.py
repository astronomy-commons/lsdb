from __future__ import annotations

from typing import TYPE_CHECKING

import hats as hc

if TYPE_CHECKING:
    from lsdb import Catalog


# pylint: disable=too-few-public-methods
class CatalogCollection:
    """A collection of HATS Catalog DataFrame to perform analysis of sky
    catalogs and efficient spatial operations.

    Attributes:
        catalog: `lsdb.Catalog` object representing the structure and metadata
            of the HATS stand-alone catalog.
        index: `hats.IndexCatalog` object representing the index catalog.
    """

    catalog: Catalog
    index: hc.catalog.index.index_catalog.IndexCatalog | None = None

    def __init__(
        self,
        catalog: Catalog,
        index: hc.catalog.index.index_catalog.IndexCatalog | None = None,
    ):
        """Initialise a CatalogCollection object.

        Not to be used to load a catalog directly, use one of the `lsdb.from_...` or
        `lsdb.load_...` methods

        Args:
            catalog: `lsdb.Catalog` object representing the structure and metadata
                of the HATS stand-alone catalog.
            index: `hats.IndexCatalog` object representing the index catalog.
        """
        self.catalog = catalog
        self.index = index
