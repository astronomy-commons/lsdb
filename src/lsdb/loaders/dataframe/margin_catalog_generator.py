from __future__ import annotations

from typing import Dict, List, Tuple

import dask.dataframe as dd
import healpy as hp
import hipscat as hc
import numpy as np
import pandas as pd
from hipscat import pixel_math
from hipscat.catalog import CatalogType
from hipscat.catalog.margin_cache import MarginCacheCatalogInfo
from hipscat.pixel_math import HealpixPixel
from hipscat.pixel_math.healpix_pixel_function import get_pixel_argsort

from lsdb import Catalog
from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.dataframe.from_dataframe_utils import (
    _format_margin_partition_dataframe,
    _generate_dask_dataframe,
)


class MarginCatalogGenerator:
    """Creates a HiPSCat formatted margin catalog"""

    def __init__(
        self,
        catalog: Catalog,
        margin_order: int | None = -1,
        margin_threshold: float = 5.0,
    ) -> None:
        """Initialize a MarginCatalogGenerator

        Args:
            catalog (Catalog): The LSDB catalog to generate margins for
            margin_order (int): The order at which to generate the margin cache
            margin_threshold (float): The size of the margin cache boundary, in arcseconds
        """
        self.dataframe = catalog.compute().copy()
        self.hc_structure = catalog.hc_structure
        self.margin_threshold = margin_threshold
        self.margin_order = self._set_margin_order(margin_order)

    def _set_margin_order(self, margin_order: int | None) -> int:
        """Calculate the order of the margin cache to be generated. If not provided
        the margin will be greater than that of the original catalog by 1.

        Args:
            margin_order (int): The order to generate the margin cache with

        Returns:
            The validated order of the margin catalog.

        Raises:
            ValueError: if the provided margin order is lower than that of the catalog.
        """
        margin_pixel_k = self.hc_structure.partition_info.get_highest_order() + 1
        if margin_order is None or margin_order == -1:
            margin_order = margin_pixel_k
        elif margin_order < margin_pixel_k:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )
        return margin_order

    def create_catalog(self) -> MarginCatalog | None:
        """Create a margin catalog for another pre-computed catalog

        Returns:
            Margin catalog object, or None if the margin is empty.
        """
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map()
        margin_pixels = list(ddf_pixel_map.keys())
        if total_rows == 0:
            return None
        margin_catalog_info = self._create_catalog_info(total_rows)
        margin_structure = hc.catalog.MarginCatalog(margin_catalog_info, margin_pixels)
        return MarginCatalog(ddf, ddf_pixel_map, margin_structure)

    def _generate_dask_df_and_map(self) -> Tuple[dd.core.DataFrame, Dict[HealpixPixel, int], int]:
        """Create the Dask Dataframe containing the data points in the margins
        for the catalog as well as the mapping of those HEALPix to Dataframes

        Returns:
            Tuple containing the Dask Dataframe, the mapping of margin HEALPix
            to the respective partitions and the total number of rows.
        """
        healpix_pixels = self.hc_structure.get_healpix_pixels()
        negative_pixels = self.hc_structure.generate_negative_tree_pixels()
        combined_pixels = healpix_pixels + negative_pixels
        margin_pairs_df = self._find_margin_pixel_pairs(combined_pixels)

        # Compute points for each margin pixels
        margins_pixel_df = self._create_margins(margin_pairs_df)
        pixels, partitions = list(margins_pixel_df.keys()), list(margins_pixel_df.values())

        # Generate pixel map ordered by _hipscat_index
        pixel_order = get_pixel_argsort(pixels)
        ordered_pixels = np.asarray(pixels)[pixel_order]
        ordered_partitions = [partitions[i] for i in pixel_order]
        ddf_pixel_map = {pixel: index for index, pixel in enumerate(ordered_pixels)}

        # Generate the dask dataframe with the pixels and partitions
        ddf, total_rows = _generate_dask_dataframe(ordered_partitions, ordered_pixels)
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
            margins = pixel_math.get_margin(order, pix, d_order)
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
        margin_pixel_df_map: Dict[HealpixPixel, pd.DataFrame] = {}
        self.dataframe["margin_pixel"] = hp.ang2pix(
            2**self.margin_order,
            self.dataframe[self.hc_structure.catalog_info.ra_column].values,
            self.dataframe[self.hc_structure.catalog_info.dec_column].values,
            lonlat=True,
            nest=True,
        )
        constrained_data = self.dataframe.reset_index().merge(margin_pairs_df, on="margin_pixel")
        if len(constrained_data):
            for partition_group, partition_df in constrained_data.groupby(
                ["partition_order", "partition_pixel"]
            ):
                margin_pixel = HealpixPixel(partition_group[0], partition_group[1])
                df = self._get_data_in_margin(partition_df, margin_pixel)
                if len(df):
                    df = _format_margin_partition_dataframe(df)
                    margin_pixel_df_map[margin_pixel] = df
        return margin_pixel_df_map

    def _get_data_in_margin(self, partition_df: pd.DataFrame, margin_pixel: HealpixPixel) -> pd.DataFrame:
        """Calculate the margin boundaries for the HEALPix and include the points
        on the margin according to the specified threshold

        Args:
            partition_df (pd.DataFrame): The margin pixel data
            margin_pixel (HealpixPixel): The margin HEALPix

        Returns:
            A Pandas Dataframe with the points of the partition that are within
            the specified threshold in the margin.
        """
        margin_mask = pixel_math.check_margin_bounds(
            partition_df[self.hc_structure.catalog_info.ra_column].values,
            partition_df[self.hc_structure.catalog_info.dec_column].values,
            margin_pixel.order,
            margin_pixel.pixel,
            self.margin_threshold,
        )
        return partition_df.iloc[margin_mask]

    def _create_catalog_info(self, total_rows: int) -> MarginCacheCatalogInfo:
        """Create the margin catalog info object

        Args:
            total_rows (int): The number of elements in the margin catalog

        Returns:
            The margin catalog info object.
        """
        catalog_name = self.hc_structure.catalog_info.catalog_name
        return MarginCacheCatalogInfo(
            catalog_name=f"{catalog_name}_margin",
            catalog_type=CatalogType.MARGIN,
            total_rows=total_rows,
            primary_catalog=catalog_name,
            margin_threshold=self.margin_threshold,
        )
