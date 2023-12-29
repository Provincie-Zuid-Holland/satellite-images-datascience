import pandas as pd


def filter_black_spots(df: pd.DataFrame) -> pd.DataFrame:
    red_zero_mask = df["red"] != 0
    green_zero_mask = df["green"] != 0
    blue_zero_mask = df["blue"] != 0
    return df[red_zero_mask & green_zero_mask & blue_zero_mask]
