from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import hats.catalog
import nested_pandas as npd
import numpy as np
from hats import HealpixPixel
from hats.catalog import TableProperties

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


class AbstractObjectAggregator(ABC):

    def validate(self, catalog: Catalog):
        pass

    def aggregate_objects(
        self,
        partition: npd.NestedFrame,
        pixel: HealpixPixel,
        properties: TableProperties,
        margin_properties: TableProperties,
        object_ids: np.ndarray,
    ) -> npd.NestedFrame:
        pass

    @abstractmethod
    def perform_object_aggregation(self) -> npd.NestedFrame:
        pass

    @abstractmethod
    def get_meta_df(self) -> npd.NestedFrame:
        pass

    def get_hc_structure(self, catalog: Catalog) -> hats.catalog.Catalog:
        pass
