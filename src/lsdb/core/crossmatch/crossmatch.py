import inspect
from typing import Type

import nested_pandas as npd
import pandas as pd

from lsdb.catalog import Catalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.loaders.dataframe.from_dataframe import from_dataframe


def crossmatch(
    left: Catalog | npd.NestedFrame | pd.DataFrame,
    right: Catalog | npd.NestedFrame | pd.DataFrame,
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
    """Perform a cross-match between two frames, a catalog and a frame, or a frame and a catalog.

    See Catalog.crossmatch for more information.

    Args:
        left (Catalog | NestedFrame): The left catalog or frame to crossmatch.
        right (Catalog | NestedFrame): The right catalog or frame to crossmatch.
        suffixes (tuple[str, str], optional): Suffixes to append to overlapping column names.
            Defaults to None.
        algorithm (Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm, optional): The
            crossmatch algorithm to use. Defaults to BuiltInCrossmatchAlgorithm.KD_TREE.
        output_catalog_name (str, optional): The name of the output catalog. Defaults to None.
        require_right_margin (bool, optional): Whether to require a right margin. Defaults to False.
        left_args (dict, optional): Additional keyword arguments to pass to from_dataframe for the
            left catalog. Will override any **kwargs given after (for left catalog only). Defaults
            to None.
        right_args (dict, optional): Additional keyword arguments to pass to from_dataframe for the
            right catalog. Will override any **kwargs given after (for right catalog only). Defaults
            to None.
        **kwargs: Additional keyword arguments to pass to Catalog.crossmatch.

    Returns:
        Catalog: The crossmatched catalog.
    """
    # Separate kwargs for from_dataframe and crossmatch.
    sig = inspect.signature(from_dataframe)
    from_dataframe_arg_names = list(sig.parameters.keys())
    from_dataframe_kwargs = {k: kwargs.pop(k) for k in from_dataframe_arg_names if k in kwargs}

    left_args = from_dataframe_kwargs | (left_args or {})
    right_args = from_dataframe_kwargs | (right_args or {})

    # Check for conflicting right margin arguments.
    if require_right_margin and right_args.get("margin_threshold") is None:
        raise ValueError("If require_right_margin is True, margin_threshold must not be None.")

    # Check if either given data set is a frame, and if so, convert it to a Catalog.
    if not isinstance(left, Catalog):
        if not isinstance(left, (pd.DataFrame, npd.NestedFrame)):
            raise TypeError(f"Left argument must be a DataFrame, NestedFrame, or Catalog, not {type(left)}.")
        left = from_dataframe(left, **left_args)

    if not isinstance(right, Catalog):
        if not isinstance(right, (pd.DataFrame, npd.NestedFrame)):
            raise TypeError(
                f"Right argument must be a DataFrame, NestedFrame, or Catalog, not {type(right)}."
            )
        right = from_dataframe(right, **right_args)

    # Call the crossmatch method with the given or newly generated Catalogs.
    return Catalog.crossmatch(
        left, right, suffixes, algorithm, output_catalog_name, require_right_margin, **kwargs
    )
