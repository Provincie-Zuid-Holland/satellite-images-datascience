import pandas as pd

from .features import FeatureType, selected_features


class DataPreprocessor:
    """ """

    def __init__(self, features_to_generate: list = selected_features):
        self.features = features_to_generate
        self.feature_to_method_map = {
            FeatureType.FRACTION_BRIGHT.value: DataPreprocessor._get_fraction_bright,
            FeatureType.FRACTION_RELATIVE_BRIGHT.value: DataPreprocessor._get_fraction_relative_bright,
            FeatureType.FRACTION_BRIGHT_FROM_MAX.value: DataPreprocessor._get_fraction_bright_from_max,
            FeatureType.FRACTION_COLOUR_BRIGHT.value: DataPreprocessor._get_fraction_colour_bright,
            FeatureType.NUMBER_BRIGHT_PIXELS.value: DataPreprocessor._get_pixels_bright,
            FeatureType.NUMBER_OF_PIXELS.value: DataPreprocessor._get_number_of_pixels,
            FeatureType.COLOUR_MEAN.value: DataPreprocessor._get_colour_mean,
            FeatureType.COLOUR_QUANTILE.value: DataPreprocessor._get_colour_quantile,
        }

    def transform(self, df: pd.DataFrame) -> pd.Series:
        output = pd.Series()
        for feature in self.features:
            output[str(feature)] = self.feature_to_method_map[
                feature.feature_type.value
            ](df=df, **feature.kwargs)
        return output

    @staticmethod
    def _get_fraction_bright(df: pd.DataFrame, minimal_brightness: int) -> float:
        return (
            (df["red"] > minimal_brightness)
            & (df["green"] > minimal_brightness)
            & (df["blue"] > minimal_brightness)
        ).mean()

    @staticmethod
    def _get_fraction_relative_bright(df: pd.DataFrame, minimal_quantile: int) -> float:
        red_quantile = df["red"].quantile(minimal_quantile)
        green_quantile = df["green"].quantile(minimal_quantile)
        blue_quantile = df["blue"].quantile(minimal_quantile)
        return (
            (df["red"] > red_quantile)
            & (df["green"] > green_quantile)
            & (df["blue"] > blue_quantile)
        ).mean()

    @staticmethod
    def _get_fraction_bright_from_max(
        df: pd.DataFrame, minimal_fraction_brightness: int
    ) -> float:
        return (
            (df["red"] > minimal_fraction_brightness * df["red"].max())
            & (df["green"] > minimal_fraction_brightness * df["green"].max())
            & (df["blue"] > minimal_fraction_brightness * df["blue"].max())
        ).mean()

    @staticmethod
    def _get_fraction_colour_bright(
        df: pd.DataFrame, minimal_brightness: int, colour: str
    ) -> float:
        return (df[colour] > minimal_brightness).mean()

    @staticmethod
    def _get_pixels_bright(df: pd.DataFrame, minimal_brightness: int) -> float:
        return (
            (df["red"] > minimal_brightness)
            & (df["green"] > minimal_brightness)
            & (df["blue"] > minimal_brightness)
        ).sum()

    @staticmethod
    def _get_number_of_pixels(df: pd.DataFrame) -> int:
        return len(df)

    @staticmethod
    def _get_colour_mean(df: pd.DataFrame, colour: str) -> float:
        return df[colour].mean()

    @staticmethod
    def _get_colour_quantile(df: pd.DataFrame, colour: str, quantile: float) -> float:
        return df[colour].quantile(quantile)
