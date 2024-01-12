import pandas as pd


def get_cross_validation_results_filepath(
    location: str, satellite_constellation: str, df: pd.DataFrame
) -> str:
    first_date = df["date"].min()
    last_date = df["date"].max()
    return f"../saved_models/{satellite_constellation}_{location}_{first_date}_to_{last_date}_cross_validation_results.pkl"


def get_model_filepath(
    location: str, satellite_constellation: str, df: pd.DataFrame
) -> str:
    first_date = df["date"].min()
    last_date = df["date"].max()
    return f"../saved_models/{satellite_constellation}_{location}_{first_date}_to_{last_date}_random_forest_classifier.sav"
