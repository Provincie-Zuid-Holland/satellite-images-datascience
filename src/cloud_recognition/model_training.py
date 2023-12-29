from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_filenames(
    df: pd.DataFrame, column_to_split_on: str = None, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into a train_df and a test_df using train_test_split method from sklearn. If column_to_split_on != None it splits using the column
    """
    if column_to_split_on:
        column_values = df[column_to_split_on].unique()
        train_values, test_values = train_test_split(column_values, **kwargs)
        train_df = df[df[column_to_split_on].isin(train_values)]
        test_df = df[df[column_to_split_on].isin(test_values)]
    else:
        train_df, test_df = train_test_split(df, **kwargs)
    return train_df, test_df
