from typing import Type

import nested_pandas as npd
import pandas as pd

from lsdb.catalog import Catalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm, is_builtin_algorithm
from lsdb.loaders.dataframe.from_dataframe import from_dataframe


def _validate_and_convert_to_catalog(
    data: npd.NestedFrame | pd.DataFrame, data_args: dict, suffix: str | None, default_suffix: str
) -> Catalog:
    """Validate arguments and convert a DataFrame or NestedFrame to a Catalog."""
    if isinstance(data, Catalog):
        return data
    if not isinstance(data, (pd.DataFrame, npd.NestedFrame)):
        raise TypeError(f"Argument must be a DataFrame, NestedFrame, or Catalog, not {type(data)}.")

    # Pick catalog name: either use the user-specified suffixes, or default to "left" or "right".
    if "catalog_name" not in data_args:
        data_args["catalog_name"] = suffix if suffix else default_suffix

    # Convert the DataFrame to a Catalog.
    data = from_dataframe(data, **data_args)
    return data


def crossmatch(
    left: Catalog | npd.NestedFrame | pd.DataFrame,
    right: Catalog | npd.NestedFrame | pd.DataFrame,
    ra_column: str | None = None,
    dec_column: str | None = None,
    suffixes: tuple[str, str] | None = None,
    algorithm: (
        Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
    ) = BuiltInCrossmatchAlgorithm.KD_TREE,
    output_catalog_name: str | None = None,
    require_right_margin: bool = False,
    left_args: dict | None = None,
    right_args: dict | None = None,
    **kwargs,
) -> Catalog:
    """Perform a cross-match between two frames, two catalogs, a catalog and a frame, or a frame and
    a catalog.

    See Catalog.crossmatch for more information on cross-matching.

    Args:
        left (Catalog | NestedFrame): The left catalog or frame to crossmatch.
        right (Catalog | NestedFrame): The right catalog or frame to crossmatch.
        ra_column (str, optional): The name of the right ascension column for both catalogs,
            if passing dataframes. Can be specified in the left_args or right_args dictionaries if
            left and right catalogs have different RA column names. Defaults to None, which will use
            the default column names "ra", "Ra", or "RA" if they exist in the DataFrame.
        dec_column (str, optional): The name of the declination column for both catalogs,
            if passing dataframes. Can be specified in the left_args or right_args dictionaries if
            left and right catalogs have different dec column names. Defaults to None, which will use
            the default column names "dec", "Dec", or "DEC" if they exist in the DataFrame.
        suffixes (tuple[str, str], optional): Suffixes to append to overlapping column names.
            Defaults to None.
        algorithm (Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm, optional): The
            crossmatch algorithm to use. Defaults to BuiltInCrossmatchAlgorithm.KD_TREE.
        output_catalog_name (str, optional): The name of the output catalog. Defaults to None.
        require_right_margin (bool, optional): Whether to require a right margin. Defaults to False.
        left_args (dict, optional): Keyword arguments to pass to from_dataframe for the left
            catalog. Defaults to None.
        right_args (dict, optional): Keyword arguments to pass to from_dataframe for the right
            catalog. Defaults to None.
        **kwargs: Additional keyword arguments to pass to Catalog.crossmatch.

    Returns:
        Catalog: The crossmatched catalog.
    """
    # Initialize dictionaries if not given.
    left_args = left_args or {}
    right_args = right_args or {}

    # Check for conflicting right margin arguments.
    if require_right_margin and right_args.get("margin_threshold") is None:
        raise ValueError("If require_right_margin is True, margin_threshold must not be None.")

    # Check if the margin should be generated according to the
    # maximum radius specified for the crossmatch.
    if is_builtin_algorithm(algorithm) and "radius_arcsec" in kwargs:
        radius_arcsec = kwargs.get("radius_arcsec")
        if "margin_threshold" not in right_args:
            right_args["margin_threshold"] = radius_arcsec

    # Update left_args and right_args with ra_column and dec_column if given.
    if ra_column:
        left_args["ra_column"] = ra_column
        right_args["ra_column"] = ra_column
    if dec_column:
        left_args["dec_column"] = dec_column
        right_args["dec_column"] = dec_column

    # Validate and convert left and right to Catalogs if necessary.
    left = _validate_and_convert_to_catalog(left, left_args, suffixes[0] if suffixes else None, "left")
    right = _validate_and_convert_to_catalog(right, right_args, suffixes[1] if suffixes else None, "right")

    # Call the crossmatch method with the given or newly generated Catalogs.
    return Catalog.crossmatch(
        left, right, suffixes, algorithm, output_catalog_name, require_right_margin, **kwargs
    )
