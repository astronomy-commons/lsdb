import nested_pandas as npd
import pandas as pd
import pyarrow as pa
from dask import delayed
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

import lsdb.nested as nd
from lsdb.dask.divisions import get_pixels_divisions


def _generate_dask_dataframe(
    pixel_dfs: list[npd.NestedFrame], pixels: list[HealpixPixel], use_pyarrow_types: bool = True
) -> tuple[nd.NestedFrame, int]:
    """Create the Dask Dataframe from the list of HEALPix pixel Dataframes

    Args:
        pixel_dfs (List[npd.NestedFrame]): The list of HEALPix pixel Dataframes
        pixels (List[HealpixPixel]): The list of HEALPix pixels in the catalog
        use_pyarrow_types (bool): If True, use pyarrow types. Defaults to True.

    Returns:
        The catalog's Dask Dataframe and its total number of rows.
    """
    pixel_dfs = [_convert_dtypes_to_pyarrow(df) for df in pixel_dfs] if use_pyarrow_types else pixel_dfs
    schema = pixel_dfs[0].iloc[:0, :].copy() if len(pixels) > 0 else []
    delayed_dfs = [delayed(df) for df in pixel_dfs]
    divisions = get_pixels_divisions(pixels)
    ddf = nd.NestedFrame.from_delayed(delayed_dfs, meta=schema, divisions=divisions)
    return ddf, len(ddf)


def _convert_dtypes_to_pyarrow(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the columns (and index) of a Pandas DataFrame to pyarrow types.

    Args:
        df (pd.DataFrame): A Pandas DataFrame

    Returns:
        A new DataFrame, with columns of pyarrow types. The return value is a
        shallow copy of the initial DataFrame to avoid copying the data.
    """
    new_series = {}
    df_index = df.index.astype(pd.ArrowDtype(pa.int64()))
    for column in df.columns:
        pa_array = pa.array(df[column], from_pandas=True)
        series = pd.Series(pa_array, dtype=pd.ArrowDtype(pa_array.type), copy=False, index=df_index)
        new_series[column] = series
    return pd.DataFrame(new_series, index=df_index, copy=False)


def _format_margin_partition_dataframe(dataframe: npd.NestedFrame) -> npd.NestedFrame:
    """Finalizes the dataframe for a margin catalog partition

    Args:
        dataframe (pd.DataFrame): The partition dataframe

    Returns:
        The dataframe for a margin partition with the data points and
        the respective pixel information.
    """
    dataframe = dataframe.drop(columns=["margin_pixel", "partition_order", "partition_pixel"])
    return dataframe.set_index(SPATIAL_INDEX_COLUMN).sort_index()


def _has_named_index(dataframe: npd.NestedFrame) -> bool:
    """Heuristic to determine if a dataframe has some meaningful index.

    This will reject dataframes with no index name for a single index,
    or empty names for multi-index (e.g. [] or [None]).
    """
    if dataframe.index.name is not None:
        ## Single index with a given name.
        return True
    if len(dataframe.index.names) == 0 or all(name is None for name in dataframe.index.names):
        return False
    return True
