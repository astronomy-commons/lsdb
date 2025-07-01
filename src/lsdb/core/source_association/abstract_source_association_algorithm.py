from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import nested_pandas as npd
import numpy as np
from hats import HealpixPixel
from hats.catalog import TableProperties

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


class AbstractSourceAssociationAlgorithm(ABC):

    object_id_type = np.int64

    def validate(self, catalog: Catalog):
        pass

    @abstractmethod
    def associate_sources(
        self,
        partition: npd.NestedFrame,
        pixel: HealpixPixel,
        properties: TableProperties,
        margin_properties: TableProperties,
        source_id_col: str,
    ) -> np.ndarray:
        pass
