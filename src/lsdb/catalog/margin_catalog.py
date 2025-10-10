from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hats as hc
import pandas as pd
from upath import UPath

import lsdb.nested as nd
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
from lsdb.types import DaskDFPixelMap

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


class MarginCatalog(HealpixDataset):
    """LSDB Catalog DataFrame to contain the "margin" of another HATS catalog.
    spatial operations.

    Attributes:
        hc_structure: `hats.MarginCatalog` object representing the structure
                      and metadata of the HATS catalog
    """

    hc_structure: hc.catalog.MarginCatalog

    def __init__(
        self,
        ddf: nd.NestedFrame,
        ddf_pixel_map: DaskDFPixelMap,
        hc_structure: hc.catalog.MarginCatalog,
        loading_config: HatsLoadingConfig | None = None,
    ):
        super().__init__(ddf, ddf_pixel_map, hc_structure, loading_config=loading_config)

    def to_hats(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        overwrite: bool = False,
        error_if_empty: bool = False,
        **kwargs,
    ):
        super().to_hats(
            base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            overwrite=overwrite,
            error_if_empty=error_if_empty,
            **kwargs,
        )

    def write_catalog(
        self,
        base_catalog_path: str | Path | UPath,
        *,
        catalog_name: str | None = None,
        default_columns: list[str] | None = None,
        overwrite: bool = False,
        error_if_empty: bool = False,
        **kwargs,
    ):
        super().write_catalog(
            base_catalog_path,
            catalog_name=catalog_name,
            default_columns=default_columns,
            overwrite=overwrite,
            error_if_empty=error_if_empty,
            **kwargs,
        )


def _validate_margin_catalog(margin_catalog: MarginCatalog, catalog: Catalog):
    """Validate that the margin and main catalogs have compatible columns and types."""
    try:
        # pylint: disable=protected-access
        pd.testing.assert_frame_equal(margin_catalog._ddf._meta, catalog._ddf._meta)
    except AssertionError as e:
        raise ValueError(
            f"The margin catalog and the main catalog must have the same schema. Schemas do not match:\n{e}"
        ) from None
    if margin_catalog.hc_structure.catalog_info.ra_column != catalog.hc_structure.catalog_info.ra_column:
        raise ValueError("RA column names do not match between margin and main catalog")
    if margin_catalog.hc_structure.catalog_info.dec_column != catalog.hc_structure.catalog_info.dec_column:
        raise ValueError("Dec column names do not match between margin and main catalog")
