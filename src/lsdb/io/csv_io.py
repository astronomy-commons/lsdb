import pandas as pd
from hipscat.io import FilePointer


def read_csv_file_to_pandas(file_pointer: FilePointer, **kwargs) -> pd.DataFrame:
    """Reads a CSV file to a Pandas DataFrame

    Args:
        file_pointer (FilePointer): File Pointer to a csv file
        **kwargs: Additional arguments to pass to dataframe read_csv method

    Returns:
        Pandas DataFrame with the data from the csv file
    """
    return pd.read_csv(file_pointer, **kwargs)
