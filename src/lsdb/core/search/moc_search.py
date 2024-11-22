from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.units as u
import nested_pandas as npd
from hats.catalog import TableProperties
from mocpy import MOC

from lsdb.core.search.abstract_search import AbstractSearch

if TYPE_CHECKING:
    from lsdb.types import HCCatalogTypeVar


class MOCSearch(AbstractSearch):
    """Filter the catalog by a MOC.

    Filters partitions in the catalog to those that are in a specified moc.
    """

    def __init__(self, moc: MOC, fine: bool = True):
        super().__init__(fine)
        self.moc = moc

    def filter_hc_catalog(self, hc_structure: HCCatalogTypeVar) -> HCCatalogTypeVar:
        return hc_structure.filter_by_moc(self.moc)

    def search_points(self, frame: npd.NestedFrame, metadata: TableProperties) -> npd.NestedFrame:
        df_ras = frame[metadata.ra_column].to_numpy()
        df_decs = frame[metadata.dec_column].to_numpy()
        mask = self.moc.contains_lonlat(df_ras * u.deg, df_decs * u.deg)
        return frame.iloc[mask]
