from __future__ import annotations

from typing import Dict, List

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
            ValueError, if the provided margin order is lower than that of the catalog.
        """
        margin_pixel_k = self.hc_structure.partition_info.get_highest_order() + 1
        if margin_order is None or margin_order == -1:
            margin_order = margin_pixel_k
        elif margin_order < margin_pixel_k:
            raise ValueError(
                "margin_order must be of a higher order than the highest order catalog partition pixel."
            )
        return margin_order

    def create_catalog(self) -> MarginCatalog:
        """Create a margin catalog for another pre-computed catalog

        Returns:
            Margin catalog object for the provided catalog
        """
        ddf, ddf_pixel_map, total_rows = self._generate_dask_df_and_map()
        margin_catalog_info = self._create_catalog_info(total_rows)
        margin_pixels = list(ddf_pixel_map.keys())
        margin_structure = hc.catalog.MarginCatalog(margin_catalog_info, margin_pixels)
        return MarginCatalog(ddf, ddf_pixel_map, margin_structure)

    def _generate_dask_df_and_map(self):
        """Create the Dask Dataframe containing the data points in the margins
        for the catalog, as well as the mapping of those HEALPix pixels to
        HEALPix Dataframes

        Returns:
            Tuple containing the Dask Dataframe, the mapping of HEALPix pixels
            to the respective Pandas Dataframes and the total number of rows.
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

    def _create_margins(self, margin_pairs_df: pd.DataFrame) -> Dict[HealpixPixel, pd.DataFrame]:
        """Compute the margins for all the pixels in the catalog

        Args:
            margin_pairs_df (pd.DataFrame): A DataFrame containing all the combinations
                of catalog pixels and respective margin pixels

        Returns:
            A dictionary that maps each margin pixel to the respective DataFrame shards
            that have points for each catalog partition.
        """
        margin_shards: Dict[HealpixPixel, List[pd.DataFrame]] = {}
        self.dataframe["margin_pixel"] = hp.ang2pix(
            2**self.margin_order,
            self.dataframe[self.hc_structure.catalog_info.ra_column].values,
            self.dataframe[self.hc_structure.catalog_info.dec_column].values,
            lonlat=True,
            nest=True,
        )
        constrained_data = self.dataframe.reset_index().merge(margin_pairs_df, on="margin_pixel")
        if len(constrained_data):
            constrained_data.groupby(["partition_order", "partition_pixel"]).apply(
                self._to_margin_shard, margin_shards
            )
        reduced_margin_shards = {
            pixel: pd.concat(shard_dfs, axis=0) for pixel, shard_dfs in margin_shards.items()
        }
        return reduced_margin_shards

    def _to_margin_shard(
        self, partition: pd.DataFrame, margin_shards: Dict[HealpixPixel, List[pd.DataFrame]]
    ):
        """Get the points of a margin pixel which are inside a catalog partition

        Args:
            partition (pd.DataFrame): Catalog partition DataFrame
            margin_shards (dict): A dictionary that maps each margin pixel and the
                respective DataFrame shards
        """
        partition_order = partition["partition_order"].iloc[0]
        partition_pixel = partition["partition_pixel"].iloc[0]
        margin_pixel = HealpixPixel(partition_order, partition_pixel)
        df = self._get_data_in_margin(partition, margin_pixel)
        if len(df):
            df = _format_margin_partition_dataframe(df)
            margin_shards.setdefault(margin_pixel, []).append(df)

    def _find_margin_pixel_pairs(self, pixels: List[HealpixPixel]) -> pd.DataFrame:
        """Calculate the pairs of catalog pixels and their margin pixels

        Args:
            pixels (List[HealpixPixel]): The HEALPix to compute margin pixels for.
                These include the catalog pixels as well as the negative pixels.

        Returns:
            A Pandas Dataframe with the many-to-many mapping between the partitions
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

    def _get_data_in_margin(self, partition_df: pd.DataFrame, pixel: HealpixPixel) -> pd.DataFrame:
        """Calculate the margin boundaries for the HEALPix and include the points
        on the margins according to the specified threshold

        Args:
            partition_df (pd.DataFrame): The partition dataframe
            pixel (HealpixPixel): The HEALPix pixel to get the margin points for

        Returns:
            A Pandas Dataframe with the points of the partition that are within
            the specified margin.
        """
        margin_mask = pixel_math.check_margin_bounds(
            partition_df[self.hc_structure.catalog_info.ra_column].values,
            partition_df[self.hc_structure.catalog_info.dec_column].values,
            pixel.order,
            pixel.pixel,
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
