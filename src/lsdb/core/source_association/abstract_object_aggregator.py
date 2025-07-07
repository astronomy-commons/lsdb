from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import hats.catalog
import nested_pandas as npd
import numpy as np
import pandas as pd
from hats import HealpixPixel
from hats.catalog import TableProperties
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, compute_spatial_index

import pyarrow as pa

from lsdb.dask.merge_catalog_functions import filter_by_spatial_index_to_pixel

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


def perform_reduction(*args, columns=None, fun=None, **kwargs):
    obj_id = args[0]
    return fun({c: a for c, a in zip(columns, args[1:])}, obj_id, **kwargs)


class AbstractObjectAggregator(ABC):

    def __init__(self, ra_name: str = "ra", dec_name: str = "dec", append_lc=True, lc_name="lc"):
        self.ra_name = ra_name
        self.dec_name = dec_name
        self.append_lc = append_lc
        self.lc_name = lc_name

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
        ndf = npd.NestedFrame.from_flat(df, base_columns=["obj_id"], name=self.lc_name)
        nested_cols = ndf[self.lc_name].nest.fields
        res = ndf.reduce(
            perform_reduction,
            *(["obj_id"] + [f"{self.lc_name}.{f}" for f in nested_cols]),
            columns=nested_cols,
            fun=self.perform_object_aggregation,
            pixel=pixel,
            properties=properties,
        )
        if self.append_lc:
            lc_ndf = ndf[[self.lc_name]]
            if SPATIAL_INDEX_COLUMN in nested_cols:
                lc = ndf[self.lc_name].nest.without_field(SPATIAL_INDEX_COLUMN)
                lc_ndf[self.lc_name] = lc
            res = pd.concat([res, lc_ndf], axis=1)
        if SPATIAL_INDEX_COLUMN not in res.columns:
            spatial_index = compute_spatial_index(res[self.ra_name], res[self.dec_name])
            pa_arr = pa.array(spatial_index)
            index_series = pd.Series(pa_arr, index=res.index, dtype=pd.ArrowDtype(pa_arr.type))
            res[SPATIAL_INDEX_COLUMN] = index_series

        res = res.set_index(SPATIAL_INDEX_COLUMN)
        res.index.name = SPATIAL_INDEX_COLUMN
        return filter_by_spatial_index_to_pixel(res, pixel.order, pixel.pixel)

    @abstractmethod
    def perform_object_aggregation(
        self, df: dict, obj_id: int, pixel: HealpixPixel = None, properties: TableProperties = None
    ) -> npd.NestedFrame:
        pass

    def get_meta_df(self, catalog: Catalog) -> npd.NestedFrame:
        object_df = self.get_object_meta_df()
        if self.append_lc:
            catalog_meta = catalog._ddf._meta
            return object_df.add_nested(catalog_meta, name="lc")
        return object_df

    @abstractmethod
    def get_object_meta_df(self) -> npd.NestedFrame:
        pass

    @abstractmethod
    def get_hc_structure(self, catalog: Catalog) -> hats.catalog.Catalog:
        pass
