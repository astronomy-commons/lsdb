from typing import Dict, List, NamedTuple, Tuple

import dask.dataframe as dd
import hipscat as hc
import pandas as pd
from dask import delayed
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.io import FilePointer, get_file_pointer_from_path
from hipscat.pixel_math import HealpixPixel, generate_histogram
from hipscat.pixel_math.hipscat_id import compute_hipscat_id, healpix_to_hipscat_id

from lsdb.catalog.catalog import Catalog, DaskDFPixelMap
from lsdb.io.csv_io import read_csv_file_to_pandas

HealpixInfo = NamedTuple("HealpixInfo", [("num_points", int), ("pixels", List[int])])


class DataframeCatalogLoader:
    """Loads a HiPSCat formatted Catalog from a Pandas Dataframe"""

    HISTOGRAM_ORDER = 10
    PARTITION_SIZE = "100MB"

    def __init__(self, path: str, threshold: int = 50, **kwargs) -> None:
        """Initializes a DataframeCatalogLoader

        Args:
            path (str): Path to a CSV file
            threshold (int): The maximum number of data points per pixel
            **kwargs: Arguments to pass to the creation of the catalog info
        """
        self._check_path_is_valid(get_file_pointer_from_path(path))
        self.path = hc.io.get_file_pointer_from_path(path)
        self.threshold = threshold
        self.catalog_info = CatalogInfo(**kwargs)

    @staticmethod
    def _check_path_is_valid(path: FilePointer):
        """Checks if pointer to CSV file is valid"""
        if not hc.io.file_io.is_regular_file(path):
            raise FileNotFoundError("Catalog file could not be found")

    def load_catalog(self) -> Catalog:
        """Load a catalog from a Pandas Dataframe, in CSV format

        Returns:
            Catalog object with data from the source given at loader initialization
        """
        df = read_csv_file_to_pandas(self.path)
        self._set_hipscat_index(df)
        pixel_map = self._get_pixel_map(df)
        ddf, ddf_pixel_map = self._load_dask_df_and_map(df, pixel_map)
        healpix_pixels = list(pixel_map.keys())
        hc_structure = self._init_hipscat_catalog(healpix_pixels)
        return Catalog(ddf, ddf_pixel_map, hc_structure)

    def _set_hipscat_index(self, df: pd.DataFrame):
        """Generates the hipscat indices for each data point
        and assigns the hipscat_index column as the dataframe index.

        Args:
            df (pd.Dataframe): The catalog Pandas Dataframe
        """
        df["hipscat_index"] = compute_hipscat_id(
            ra_values=df[self.catalog_info.ra_column],
            dec_values=df[self.catalog_info.dec_column],
        )
        df.set_index("hipscat_index", inplace=True)

    def _get_pixel_map(self, df: pd.DataFrame) -> Dict[HealpixPixel, HealpixInfo]:
        """Compute object histogram and generate the mapping between
        Healpix pixels and the respective original pixel information

        Args:
            df (pd.Dataframe): The catalog Pandas Dataframe

        Returns:
            A dictionary mapping each Healpix pixel to the respective
            information tuple. The first value of the tuple is the number
            of objects in the Healpix pixel, the second is the list of pixels
        """
        raw_histogram = generate_histogram(
            df,
            highest_order=self.HISTOGRAM_ORDER,
            ra_column=self.catalog_info.ra_column,
            dec_column=self.catalog_info.dec_column,
        )
        return hc.pixel_math.compute_pixel_map(
            raw_histogram, highest_order=self.HISTOGRAM_ORDER, threshold=self.threshold
        )

    def _load_dask_df_and_map(
        self, df: pd.DataFrame, pixel_map: Dict[HealpixPixel, HealpixInfo]
    ) -> Tuple[dd.DataFrame, DaskDFPixelMap]:
        """Load Dask DataFrame from Healpix pixel Dataframes and
        generate a mapping of Healpix pixels to Healpix Dataframes

        Args:
            df (pd.Dataframe): The catalog Pandas Dataframe
            pixel_map (Dict[HealpixPixel, HealpixInfo]): The mapping between
                HealPix pixels and respective data information

        Returns:
            Tuple containing the Dask Dataframe and the mapping of
            Healpix pixels to the respective Pandas Dataframes
        """
        # Dataframes for each destination Healpix pixel
        pixel_dfs: List[pd.DataFrame] = []
        # Mapping Healpix pixels to the respective Dataframe indices
        ddf_pixel_map: Dict[HealpixPixel, int] = {}

        for hp_pixel_index, hp_pixel_info in enumerate(pixel_map.items()):
            hp_pixel, (_, pixels) = hp_pixel_info
            # Obtain Dataframe for the current Healpix pixel
            pixel_dfs.append(self._get_dataframe_for_healpix(df, pixels))
            ddf_pixel_map[hp_pixel] = hp_pixel_index

        # Generate Dask Dataframe with original schema
        schema = pd.DataFrame(columns=df.columns).astype(df.dtypes)
        ddf = self._generate_dask_dataframe(pixel_dfs, schema)

        return ddf, ddf_pixel_map

    def _generate_dask_dataframe(self, pixel_dfs: List[pd.DataFrame], schema: pd.DataFrame) -> dd.DataFrame:
        """Create the Dask Dataframe from the list of Healpix pixel Dataframes

        Args:
            pixel_dfs (List[pd.DataFrame]): The list of Healpix pixel Dataframes
            schema (pd.Dataframe): The original Dataframe schema

        Returns:
            The catalog's Dask Dataframe
        """
        delayed_dfs = [delayed(pd.DataFrame)(df) for df in pixel_dfs]
        return dd.from_delayed(delayed_dfs, meta=schema).repartition(self.PARTITION_SIZE)

    def _init_hipscat_catalog(self, pixels: List[HealpixPixel]) -> hc.catalog.Catalog:
        """Initializes the Hipscat Catalog object

        Args:
            pixels (List[HealpixPixel]): The list of Healpix pixels

        Returns:
            The Hipscat catalog object
        """
        return hc.catalog.Catalog(self.catalog_info, pixels)

    def _get_dataframe_for_healpix(self, df: pd.DataFrame, pixels: List[int]) -> pd.DataFrame:
        """Computes the Pandas Dataframe containing the data points for a certain HealPix pixel

        Args:
            df (pd.Dataframe): The catalog Pandas Dataframe
            pixels (List[int]): The indices of the pixels inside the Healpix pixel

        Returns:
            The Pandas Dataframe containing the data points for the Healpix pixel
        """
        left_bound = healpix_to_hipscat_id(self.HISTOGRAM_ORDER, pixels[0])
        right_bound = healpix_to_hipscat_id(self.HISTOGRAM_ORDER, pixels[-1] + 1)
        return df.loc[(df.index >= left_bound) & (df.index < right_bound)]
