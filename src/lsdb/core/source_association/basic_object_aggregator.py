import hats
import nested_pandas as npd
import pandas as pd
from hats import HealpixPixel
from hats.catalog import TableProperties, CatalogType

import pyarrow as pa
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

from lsdb import Catalog
from lsdb.core.source_association.abstract_object_aggregator import AbstractObjectAggregator


class BasicObjectAggregator(AbstractObjectAggregator):

    def __init__(self, properties: TableProperties, ra_name="ra", dec_name="dec"):
        self.properties = properties
        self.ra_name = ra_name
        self.dec_name = dec_name

    def get_meta_df(self) -> npd.NestedFrame:
        meta = npd.NestedFrame(
            {
                "object_id": pd.Series([], dtype=pd.ArrowDtype(pa.int64())),
                self.ra_name: pd.Series([], dtype=pd.ArrowDtype(pa.float64())),
                self.dec_name: pd.Series([], dtype=pd.ArrowDtype(pa.float64())),
            }
        )
        meta.index.name = SPATIAL_INDEX_COLUMN
        return meta

    def get_hc_structure(self, catalog: Catalog) -> hats.catalog.Catalog:
        properties = hats.catalog.TableProperties(
            catalog_name=catalog.hc_structure.catalog_name + "_objects",
            catalog_type=CatalogType.OBJECT,
            ra_column=self.ra_name,
            dec_column=self.dec_name,
            hats_nrows=0,
        )
        return hats.catalog.Catalog(
            catalog_info=properties,
            pixels=catalog.hc_structure.pixel_tree,
            catalog_path=None,
            moc=catalog.hc_structure.moc,
            schema=pa.Schema.from_pandas(self.get_meta_df()).remove_metadata(),
        )

    def perform_object_aggregation(
        self, df: dict, obj_id: int, pixel: HealpixPixel = None, properties: TableProperties = None
    ) -> dict:
        return {
            "object_id": obj_id,
            self.ra_name: df[properties.ra_column][0],
            self.dec_name: df[properties.dec_column][0],
            SPATIAL_INDEX_COLUMN: df[SPATIAL_INDEX_COLUMN][0],
        }
