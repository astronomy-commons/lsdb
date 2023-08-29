from abc import ABC, abstractmethod
from typing import Tuple

import hipscat as hc

import pandas as pd


class AbstractCrossmatchAlgorithm(ABC):
    """Abstract class used to write a crossmatch algorithm"""

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_order: int,
        left_pixel: int,
        right_order: int,
        right_pixel: int,
        left_metadata: hc.catalog.Catalog,
        right_metadata: hc.catalog.Catalog,
        suffixes: Tuple[str, str],
    ):
        """Initializes a crossmatch algorithm

        Args:
            left (pd.DataFrame): Data from the pixel in the left tree
            right (pd.DataFrame): Data from the pixel in the right tree
            left_order (int): The HEALPix order of the left pixel
            left_pixel (int): The HEALPix pixel number in NESTED ordering of the left pixel
            right_order (int): The HEALPix order of the right pixel
            right_pixel (int): The HEALPix pixel number in NESTED ordering of the right pixel
            left_metadata (hipscat.Catalog): The hipscat Catalog object with the metadata of the
                left catalog
            right_metadata (hipscat.Catalog): The hipscat Catalog object with the metadata of the
                right catalog
            suffixes (Tuple[str,str]): A pair of suffixes to be appended to the end of each column
                name, with the first appended to the left columns and the second to the right
                columns
        """
        self.left = left.copy(deep=False)
        self.right = right.copy(deep=False)
        self.left_order = left_order
        self.left_pixel = left_pixel
        self.right_order = right_order
        self.right_pixel = right_pixel
        self.left_metadata = left_metadata
        self.right_metadata = right_metadata
        self.suffixes = suffixes

    @abstractmethod
    def crossmatch(self) -> pd.DataFrame:
        """Perform a crossmatch"""

    @staticmethod
    def _rename_columns_with_suffix(dataframe, suffix):
        columns_renamed = {name: name + suffix for name in dataframe.columns}
        dataframe.rename(columns=columns_renamed, inplace=True)
