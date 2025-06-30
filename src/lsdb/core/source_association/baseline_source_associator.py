import nested_pandas as npd
import numpy as np
from hats import HealpixPixel
from hats.catalog import TableProperties

from lsdb.core.source_association.abstract_source_association_algorithm import (
    AbstractSourceAssociationAlgorithm,
)


class BaselineSourceAssociationAlgorithm(AbstractSourceAssociationAlgorithm):

    def associate_sources(
        self,
        partition: npd.NestedFrame,
        pixel: HealpixPixel,
        properties: TableProperties,
        margin_properties: TableProperties,
    ) -> np.ndarray:
        return np.arange(len(partition))
