import inspect
from typing import Type

import nested_pandas as npd

from lsdb.catalog import Catalog
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_algorithms import BuiltInCrossmatchAlgorithm
from lsdb.loaders.dataframe.from_dataframe import from_dataframe


def crossmatch(
    left: Catalog | npd.NestedFrame,
    right: Catalog | npd.NestedFrame,
    suffixes: tuple[str, str] | None = None,
    algorithm: (
        Type[AbstractCrossmatchAlgorithm] | BuiltInCrossmatchAlgorithm
    ) = BuiltInCrossmatchAlgorithm.KD_TREE,
    output_catalog_name: str | None = None,
    require_right_margin: bool = False,
    **kwargs,
) -> Catalog:
    """Perform a cross-match between two frames, or a catalog and a frame (either order).

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
        **kwargs: Additional keyword arguments to pass to Catalog.crossmatch.

    Returns:
        Catalog: The crossmatched catalog.
    """
    # Separate kwargs for from_dataframe and crossmatch
    sig = inspect.signature(from_dataframe)
    from_dataframe_arg_names = list(sig.parameters.keys())
    from_dataframe_kwargs = {k: kwargs.pop(k) for k in from_dataframe_arg_names if k in kwargs}

    # Check if the left dataframe is a NestedFrame, and if so, convert it to a Catalog.
    if not isinstance(left, Catalog):
        if not isinstance(left, npd.NestedFrame):
            raise TypeError(f"Left argument must be a NestedFrame or Catalog, not {type(left)}.")
        # Convert the left DataFrame to a Catalog.
        left = from_dataframe(left, **from_dataframe_kwargs)

    # Check if the right dataframe is a NestedFrame, and if so, convert it to a Catalog.
    if not isinstance(right, Catalog):
        if not isinstance(right, npd.NestedFrame):
            raise TypeError(f"Right argument must be a NestedFrame or Catalog, not {type(right)}.")
        # Convert the right DataFrame to a Catalog.

        # TODO: consider how require_right_margin should be handled here.
        right = from_dataframe(right, **from_dataframe_kwargs)

    # Call the crossmatch method with the newly generated Catalog.
    return Catalog.crossmatch(
        left, right, suffixes, algorithm, output_catalog_name, require_right_margin, **kwargs
    )
