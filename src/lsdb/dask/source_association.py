from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hats import HealpixPixel
from hats.catalog import TableProperties
import pyarrow as pa

from lsdb.core.source_association.abstract_object_aggregator import AbstractObjectAggregator
from lsdb.core.source_association.abstract_source_association_algorithm import (
    AbstractSourceAssociationAlgorithm,
)

import nested_pandas as npd

import lsdb.nested as nd
from lsdb.dask.divisions import get_pixels_divisions
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    concat_partition_and_margin,
    filter_by_spatial_index_to_pixel,
)
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb import Catalog


def perform_source_association(
    df: npd.NestedFrame,
    margin_df: npd.NestedFrame | None,
    pix: HealpixPixel,
    margin_pix: HealpixPixel,
    properties: TableProperties,
    margin_properties: TableProperties,
    association_algorithm: AbstractSourceAssociationAlgorithm,
    source_id_column: str,
    aggregator: AbstractObjectAggregator = None,
    object_id_column_name: str = "object_id",
    healpix_id_bits: int = 10,
) -> npd.NestedFrame:
    joined_df = concat_partition_and_margin(df, margin_df)
    object_ids = association_algorithm.associate_sources(
        joined_df, pix, properties, margin_properties, source_id_column
    )
    if aggregator is None:
        # _, id_inds, id_inv = np.unique(object_ids, return_index=True, return_inverse=True)
        # obj_id_from_source_id = joined_df[source_id_column].to_numpy()[id_inds][id_inv]
        obj_id_arr = pa.array(object_ids)
        resulting_df = joined_df.assign(
            **{
                object_id_column_name: pd.Series(
                    obj_id_arr, index=joined_df.index, dtype=pd.ArrowDtype(obj_id_arr.type)
                )
            }
        )
        return filter_by_spatial_index_to_pixel(resulting_df, pix.order, pix.pixel)
    else:
        return aggregator.aggregate_objects(joined_df, pix, properties, margin_properties, object_ids)


def generate_associated_meta(
    catalog: Catalog,
    source_association_algorithm: AbstractSourceAssociationAlgorithm,
    object_aggregator: AbstractObjectAggregator = None,
    object_id_column_name: str = "object_id",
):
    if object_aggregator is not None:
        return object_aggregator.get_meta_df()
    else:
        obj_id_type = source_association_algorithm.object_id_type
        obj_id_series = pd.Series(dtype=pd.ArrowDtype(pa.from_numpy_dtype(obj_id_type)))
        return catalog._ddf._meta.assign(**{object_id_column_name: obj_id_series})


def associate_sources(
    catalog: Catalog,
    source_association_algorithm: AbstractSourceAssociationAlgorithm,
    source_id_col: str,
    object_aggregator: AbstractObjectAggregator = None,
    object_id_column_name: str = "object_id",
    healpix_id_bits: int = 10,
) -> tuple[nd.NestedFrame, DaskDFPixelMap]:
    if catalog.margin is None:
        warnings.warn(
            "Right catalog does not have a margin cache. Results may be incomplete and/or inaccurate.",
            RuntimeWarning,
        )

    source_association_algorithm.validate(catalog)
    if object_aggregator is not None:
        object_aggregator.validate(catalog)

    pixels = catalog.get_healpix_pixels()
    partitions = align_and_apply(
        [(catalog, pixels), (catalog.margin, pixels)],
        perform_source_association,
        source_association_algorithm,
        source_id_col,
        object_aggregator,
        object_id_column_name,
        healpix_id_bits=healpix_id_bits,
    )

    partition_map = {}
    for i, p in enumerate(pixels):
        partition_map[p] = i
    divisions = get_pixels_divisions(pixels)
    meta_df = generate_associated_meta(
        catalog, source_association_algorithm, object_aggregator, object_id_column_name
    )
    ddf = nd.NestedFrame.from_delayed(partitions, meta=meta_df, divisions=divisions, verify_meta=True)
    return ddf, partition_map
