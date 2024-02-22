import healpy as hp
import hipscat as hc
import pandas as pd
from hipscat import pixel_math
from hipscat.catalog import CatalogType
from hipscat.catalog.margin_cache import MarginCacheCatalogInfo
from hipscat.pixel_math import HealpixPixel

from lsdb.catalog.margin_catalog import MarginCatalog
from lsdb.loaders.dataframe.dataframe_importer import DataframeImporter


class MarginGenerator(DataframeImporter):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        hc_structure: hc.catalog.Catalog,
        margin_order: int | None = -1,
        margin_threshold: float = 5.0,
    ) -> None:
        self.dataframe = dataframe
        self.hc_structure = hc_structure
        self.margin_threshold = margin_threshold
        self.margin_order = self._calculate_margin_order(margin_order)

    def _calculate_margin_order(self, margin_order):
        """Sets the order of the margin cache to be generated.
        If not provided, the margin will be of an order that
        is higher than that of the original catalog by 1"""
        highest_order = self.hc_structure.partition_info.get_highest_order()
        margin_pixel_k = highest_order + 1
        if margin_order > -1:
            if margin_order < margin_pixel_k:
                raise ValueError(
                    "margin_order must be of a higher order "
                    "than the highest order catalog partition pixel."
                )
        else:
            margin_order = margin_pixel_k
        return margin_order

    def create_margin_catalog(self) -> MarginCatalog:
        """Creates the margin catalog"""
        ddf, ddf_pixel_map, total_rows = self._generate_margin_dask_df_and_map()
        margin_catalog_info = MarginCacheCatalogInfo(
            catalog_name=f"{self.hc_structure.catalog_info.catalog_name}_margin",
            catalog_type=CatalogType.MARGIN,
            total_rows=total_rows,
            primary_catalog=self.hc_structure.catalog_info.catalog_name,
            margin_threshold=self.margin_threshold,
        )
        margin_pixels = list(ddf_pixel_map.keys())
        margin_structure = hc.catalog.MarginCatalog(margin_catalog_info, margin_pixels)
        return MarginCatalog(ddf, ddf_pixel_map, margin_structure)

    def _generate_margin_dask_df_and_map(self):
        # Find the margin pairs of pixels for the catalog
        healpix_pixels = self.hc_structure.get_healpix_pixels()
        negative_pixels = self.hc_structure.generate_negative_tree_pixels()
        combined_pixels = healpix_pixels + negative_pixels
        margin_pairs_df = self._find_margin_pairs(combined_pixels)

        # Find in which pixels the data is located in the margin catalog
        self.dataframe["margin_pixel"] = hp.ang2pix(
            2**self.margin_order,
            self.dataframe[self.hc_structure.catalog_info.ra_column].values,
            self.dataframe[self.hc_structure.catalog_info.dec_column].values,
            lonlat=True,
            nest=True,
        )
        constrained_data = self.dataframe.merge(margin_pairs_df, on="margin_pixel")

        pixel_dfs = []
        ddf_pixel_map = {}

        # For each partition, filter the data according to the threshold
        partition_dfs = constrained_data.groupby(["partition_order", "partition_pixel"])

        for i, (_, partition) in enumerate(partition_dfs):
            order = partition["partition_order"].iloc[0]
            pix = partition["partition_pixel"].iloc[0]
            pixel = HealpixPixel(order, pix)
            df = self._get_partition_data_in_margin(partition, pixel)
            pixel_dfs.append(self._finalize_margin_partition_dataframe(df))
            ddf_pixel_map[pixel] = i

        # Generate Dask Dataframe with original schema
        ddf, total_rows = self._generate_dask_dataframe(pixel_dfs, ddf_pixel_map)
        return ddf, ddf_pixel_map, total_rows

    def _find_margin_pairs(self, pixels):
        """Calculates the pairs of catalog pixels and their margin pixels"""
        norders = []
        part_pix = []
        margin_pix = []

        for pixel in pixels:
            order = pixel.order
            pix = pixel.pixel
            d_order = self.margin_order - order
            margins = pixel_math.get_margin(order, pix, d_order)
            for m_p in margins:
                norders.append(order)
                part_pix.append(pix)
                margin_pix.append(m_p)

        return pd.DataFrame(
            zip(norders, part_pix, margin_pix),
            columns=["partition_order", "partition_pixel", "margin_pixel"],
        )

    def _get_partition_data_in_margin(self, partition_df, pixel):
        """Calculates the margin boundaries for the HEALpix and filters out
        the points of the partition according to the threshold"""
        margin_mask = pixel_math.check_margin_bounds(
            partition_df[self.hc_structure.catalog_info.ra_column].values,
            partition_df[self.hc_structure.catalog_info.dec_column].values,
            pixel.order,
            pixel.pixel,
            self.margin_threshold,
        )
        filtered_df = partition_df.loc[margin_mask]
        return self._append_partition_information_to_dataframe(filtered_df, pixel)
