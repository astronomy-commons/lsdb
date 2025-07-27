import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from hats import HealpixPixel
from hats.catalog import TableProperties
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

from lsdb.core.source_association.abstract_object_aggregator import AbstractObjectAggregator


class BasicObjectAggregator(AbstractObjectAggregator):
    def __init__(
        self,
        properties: TableProperties,
        ra_name="ra",
        dec_name="dec",
        append_lc=True,
        lc_name="lc",
        exposure_name="exposure",
    ):
        super().__init__(ra_name, dec_name, append_lc, lc_name)
        self.exposure_name = exposure_name
        self.properties = properties

    def get_object_meta_df(self) -> npd.NestedFrame:
        meta = npd.NestedFrame(
            {
                "object_id": pd.Series([], dtype=pd.ArrowDtype(pa.int64())),
                self.ra_name: pd.Series([], dtype=pd.ArrowDtype(pa.float64())),
                self.dec_name: pd.Series([], dtype=pd.ArrowDtype(pa.float64())),
            }
        )
        meta.index.name = SPATIAL_INDEX_COLUMN
        return meta

    def perform_object_aggregation(
        self, column_dict: dict, obj_id: int, pixel: HealpixPixel = None, properties: TableProperties = None
    ) -> dict:
        idx = np.argmin(column_dict[self.exposure_name])
        return {
            "object_id": obj_id,
            self.ra_name: column_dict[properties.ra_column][idx],
            self.dec_name: column_dict[properties.dec_column][idx],
            SPATIAL_INDEX_COLUMN: column_dict[SPATIAL_INDEX_COLUMN][idx],
        }
