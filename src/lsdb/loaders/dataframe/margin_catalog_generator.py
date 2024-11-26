from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import hats as hc
import hats.pixel_math.healpix_shim as hp
import nested_dask as nd
import nested_pandas as npd
import numpy as np
import pandas as pd
from hats.catalog import CatalogType, TableProperties
from hats.pixel_math import HealpixPixel, get_margin
from hats.pixel_math.healpix_pixel_function import get_pixel_argsort

from lsdb import Catalog
from lsdb.catalog.margin_catalog import MarginCatalog, _create_margin_schema
from lsdb.loaders.dataframe.from_dataframe_utils import (
    _extra_property_dict,
    _format_margin_partition_dataframe,
    _generate_dask_dataframe,
)


class MarginCatalogGenerator:
    """Creates a HATS formatted margin catalog"""

    def __init__(
        self,
        catalog: Catalog,
        margin_order: int = -1,
        margin_threshold: float | None = 5.0,
        use_pyarrow_types: bool = True,
        **kwargs,
    ) -> None:
        """Initialize a MarginCatalogGenerator

        Args:
            catalog (Catalog): The LSDB catalog to generate margins for
            margin_order (int): The order at which to generate the margin cache
            margin_threshold (float): The size of the margin cache boundary, in arcseconds
            use_pyarrow_types (bool): If True, use pyarrow types. Defaults to True.
            **kwargs: Arguments to pass to the creation of the catalog info.
        """
        self.dataframe: npd.NestedFrame = catalog.compute().copy()
        self.hc_structure = catalog.hc_structure
        self.margin_threshold = margin_threshold
        self.margin_order = margin_order
        self.use_pyarrow_types = use_pyarrow_types
        self.catalog_info_kwargs = kwargs
        self.margin_schema = _create_margin_schema(catalog.hc_structure.schema)

    def _resolve_margin_order(self):
        """Calculate the order of the margin cache to be generated. If not provided
        the margin will be calculated based on the smallest pixel possible for the threshold.

        Raises:
            ValueError: if the margin order and thresholds are incompatible with the catalog.
        """

        highest_order = int(self.hc_structure.partition_info.get_highest_order())

        if self.margin_order < 0:
            self.margin_order = hp.margin2order(margin_thr_arcmin=self.margin_threshold / 60.0)
        else:
            self.margin_threshold = hp.order2mindist(self.margin_order) * 60.0
            warnings.warn("Ignoring margin_threshold because margin_order was specified.", RuntimeWarning)

        if self.margin_order < highest_order + 1:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )

        margin_pixel_mindist = hp.order2mindist(self.margin_order)
        if margin_pixel_mindist * 60.0 < self.margin_threshold:
            raise ValueError("margin pixels must be larger than margin_threshold")

    def create_catalog(self) -> MarginCatalog | None:
        """Create a margin catalog for another pre-computed catalog.

        Only one of margin order / threshold can be specified. If the margin order
        is not specified: if the threshold is zero the margin is an empty catalog;
        if the threshold is None, the margin is not generated (it is None).

        Returns:
            Margin catalog object or None if the margin is not generated.
        """
        if self.margin_order < 0:
            if self.margin_threshold is None:
                return None
            if self.margin_threshold < 0:
                raise ValueError("margin_threshold must be positive.")
            if self.margin_threshold == 0:
                return self._create_empty_catalog()
        return self._create_catalog()

    def _create_catalog(self) -> MarginCatalog:
        """Create a non-empty margin catalog"""
        self._resolve_margin_order()
        pixels, partitions = self._get_margins()
        if len(pixels) == 0:
            return self._create_empty_catalog()
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map(pixels, partitions)
        catalog_info = self._create_catalog_info(**self.catalog_info_kwargs, total_rows=total_rows)
        margin_pixels = list(ddf_pixel_map.keys())
        margin_structure = hc.catalog.MarginCatalog(catalog_info, margin_pixels, schema=self.margin_schema)
        return MarginCatalog(ddf, ddf_pixel_map, margin_structure)

    def _create_empty_catalog(self) -> MarginCatalog:
        """Create an empty margin catalog"""
        dask_meta_schema = self.margin_schema.empty_table().to_pandas()
        ddf = nd.NestedFrame.from_pandas(dask_meta_schema, npartitions=1)
        catalog_info = self._create_catalog_info(**self.catalog_info_kwargs, total_rows=0)
        margin_structure = hc.catalog.MarginCatalog(catalog_info, [], schema=self.margin_schema)
        return MarginCatalog(ddf, {}, margin_structure)

    def _get_margins(self) -> Tuple[List[HealpixPixel], List[npd.NestedFrame]]:
        """Generates the list of pixels that have margin data, and the dataframes with the margin data for
        each partition

        Returns:
            A tuple of the list of HealpixPixels corresponding to partitions that have margin data, and
            a list of the dataframes with the margin data for each partition.
        """
        combined_pixels = (
            self.hc_structure.get_healpix_pixels() + self.hc_structure.generate_negative_tree_pixels()
        )
        margin_pairs_df = self._find_margin_pixel_pairs(combined_pixels)
        margins_pixel_df = self._create_margins(margin_pairs_df)
        pixels, partitions = list(margins_pixel_df.keys()), list(margins_pixel_df.values())
        return pixels, partitions

    def _generate_dask_df_and_map(
        self, pixels: List[HealpixPixel], partitions: List[pd.DataFrame]
    ) -> Tuple[nd.NestedFrame, Dict[HealpixPixel, int], int]:
        """Create the Dask Dataframe containing the data points in the margins
        for the catalog as well as the mapping of those HEALPix to Dataframes

        Args:
            pixels (List[HealpixPixel]): The list of healpix pixels in the catalog with margins
            partitions (List[pd.DataFrame]): The list of dataframes containing the margin rows for each
                partition, aligned with the pixels list

        Returns:
            Tuple containing the Dask Dataframe, the mapping of margin HEALPix
            to the respective partitions and the total number of rows.
        """
        # Generate pixel map ordered by _healpix_29
        pixel_order = get_pixel_argsort(pixels)
        ordered_pixels = np.asarray(pixels)[pixel_order]
        ordered_partitions = [partitions[i] for i in pixel_order]
        ddf_pixel_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}
        # Generate the dask dataframe with the pixels and partitions
        ddf, total_rows = _generate_dask_dataframe(ordered_partitions, ordered_pixels, self.use_pyarrow_types)
        return ddf, ddf_pixel_map, total_rows

    def _find_margin_pixel_pairs(self, pixels: List[HealpixPixel]) -> pd.DataFrame:
        """Calculate the pairs of catalog pixels and their margin pixels

        Args:
            pixels (List[HealpixPixel]): The list of HEALPix to compute margin pixels for.
                These include the catalog pixels as well as the negative pixels.

        Returns:
            A Pandas Dataframe with the many-to-many mapping between each catalog HEALPix
            and the respective margin pixels.
        """
        n_orders = []
        part_pix = []
        margin_pix = []

        for pixel in pixels:
            order = pixel.order
            pix = pixel.pixel
            d_order = self.margin_order - order
            margins = get_margin(order, pix, d_order)
            for m_p in margins:
                n_orders.append(order)
                part_pix.append(pix)
                margin_pix.append(m_p)

        return pd.DataFrame(
            zip(n_orders, part_pix, margin_pix),
            columns=["partition_order", "partition_pixel", "margin_pixel"],
        )

    def _create_margins(self, margin_pairs_df: pd.DataFrame) -> Dict[HealpixPixel, pd.DataFrame]:
        """Compute the margins for all the pixels in the catalog

        Args:
            margin_pairs_df (pd.DataFrame): A DataFrame containing all the combinations
                of catalog pixels and respective margin pixels

        Returns:
            A dictionary mapping each margin pixel to the respective DataFrame.
        """
        margin_pixel_df_map: Dict[HealpixPixel, npd.NestedFrame] = {}
        self.dataframe["margin_pixel"] = hp.radec2pix(
            self.margin_order,
            self.dataframe[self.hc_structure.catalog_info.ra_column].to_numpy(),
            self.dataframe[self.hc_structure.catalog_info.dec_column].to_numpy(),
        )
        constrained_data = self.dataframe.reset_index().merge(margin_pairs_df, on="margin_pixel")
        if len(constrained_data):
            for partition_group, partition_df in constrained_data.groupby(
                ["partition_order", "partition_pixel"]
            ):
                margin_pixel = HealpixPixel(partition_group[0], partition_group[1])
                margin_pixel_df_map[margin_pixel] = _format_margin_partition_dataframe(partition_df)
        return margin_pixel_df_map

    def _create_catalog_info(self, catalog_name: str | None = None, **kwargs) -> TableProperties:
        """Create the margin catalog info object

        Args:
            catalog_name (str): name of the PRIMARY catalog being created. this margin
                catalog will take on a name like `<catalog_name>_margin`.
            **kwargs: Arguments to pass to the creation of the catalog info.

        Returns:
            The margin catalog info object.
        """
        if kwargs is None:
            kwargs = {}
        kwargs.pop("catalog_type", None)
        kwargs = kwargs | _extra_property_dict(0)
        if not catalog_name:
            catalog_name = self.hc_structure.catalog_info.catalog_name

        return TableProperties(
            catalog_name=f"{catalog_name}_margin",
            catalog_type=CatalogType.MARGIN,
            ra_column=self.hc_structure.catalog_info.ra_column,
            dec_column=self.hc_structure.catalog_info.dec_column,
            primary_catalog=catalog_name,
            margin_threshold=self.margin_threshold,
            **kwargs,
        )
