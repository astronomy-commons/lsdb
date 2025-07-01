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


def perform_reduction(*args, columns=None, fun=None, **kwargs):
    obj_id = args[0]
    return fun({c: a for c, a in zip(columns, args[1:])}, obj_id, **kwargs)


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
        df = partition.reset_index(drop=False)
        df["obj_id"] = object_ids
        df.index = object_ids
        ndf = npd.NestedFrame.from_flat(df, base_columns=["obj_id"], name="nested")
        return ndf.reduce(
            perform_reduction,
            *(["obj_id"] + [f"nested.{f}" for f in ndf["nested"].nest.fields]),
            columns=ndf["nested"].nest.fields,
            fun=self.perform_object_aggregation,
            pixel=pixel,
            properties=properties,
        )

    @abstractmethod
    def perform_object_aggregation(
        self, df: dict, obj_id: int, pixel: HealpixPixel = None, properties: TableProperties = None
    ) -> npd.NestedFrame:
        pass

    @abstractmethod
    def get_meta_df(self) -> npd.NestedFrame:
        pass

    @abstractmethod
    def get_hc_structure(self, catalog: Catalog) -> hats.catalog.Catalog:
        pass
