from enum import Enum
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Location(Enum):
    COEPELDUYNEN = "coepelduynen"
    DUINENGOEREEKWADEHOEK = "Duinen Goeree & Kwade Hoek"
    VOORNESDUIN = "Voornes Duin"


class ModelType(Enum):
    BASELINE = "baseline"
    LOCATION_LINEAR_MODEL = "location_linear_model"
    LINEAR_MODEL = "linear_model"


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


class Natura2000CloudDetectionModel:
    """
    Wraps sklearn models per location & a Baseline model to allow easy application to Natura2000 satellite images for cloud detection purposes
    """

    def __init__(
        self,
        model_type: ModelType,
        locations: Union[List[Location], None] = None,
        linear_models: Union[Dict[Location, ClassifierMixin], ClassifierMixin] = None,
        pca_n_components: Union[Dict[Location, int], int] = None,
    ):
        """
        locations: list of Location objects, None if not required
        linear_models: dict where keys are Location objects and values are sklearn models (not initialised), 1 sklearn model if locations = None
        pca_n_components: dict where keys are Location objects and values are int, 1 int value if locations = None
        """
        self.model_type = model_type
        self.locations = locations
        if model_type.value == ModelType.LOCATION_LINEAR_MODEL.value:
            self.scalers = {loc: StandardScaler() for loc in self.locations}
            self.pcas = {
                loc: PCA(n_components=pca_n_components[loc]) for loc in self.locations
            }
            self.models = linear_models
        elif model_type.value == ModelType.LINEAR_MODEL.value:
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=pca_n_components)
            self.model = linear_models

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicts presence of clouds (0/1) for given df, where each row corresponds to an image that has been flattened and preprocessed by DataPreprocessor.
        """

        if self.model_type.value == ModelType.LOCATION_LINEAR_MODEL.value:
            outputs = []
            for loc in self.locations:
                location_mask = df["location"] == loc.value
                location_output = self.predict_location_linear_model(
                    df[location_mask], location=loc
                )
                outputs += [location_output]
            output = pd.concat(outputs)
        elif self.model_type.value == ModelType.BASELINE.value:
            output = self.predict_baseline(df)
        elif self.model_type.value == ModelType.LINEAR_MODEL.value:
            output = self.predict_linear_model(df)

        return output.loc[df.index]

    def predict_baseline(self, df: pd.DataFrame) -> pd.Series:
        """
        Chooses majority class
        """
        return pd.Series([False] * len(df), index=df.index)

    def predict_location_linear_model(
        self, df: pd.DataFrame, location: Location
    ) -> pd.Series:
        """
        Uses location specific linear model from self.models to predict y
        """
        if len(df) > 0:
            location_df = df.drop("location", axis=1)
            location_df = self.scalers[location].transform(location_df)
            location_df = self.pcas[location].transform(location_df)
            return pd.Series(self.models[location].predict(location_df), index=df.index)
        else:
            return pd.Series()

    def predict_linear_model(self, df: pd.DataFrame) -> pd.Series:
        """
        Uses linear self.model to predict y
        """
        input_df = df.drop("location", axis=1)
        input_df = self.scaler.transform(input_df)
        input_df = self.pca.transform(input_df)
        return pd.Series(self.model.predict(input_df), index=df.index)

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fits the models for each of the locations in self.locations
        """
        # print(self.model_type.value)
        if self.model_type.value == ModelType.BASELINE.value:
            Exception("Baseline model does not require fitting")
        elif self.model_type.value == ModelType.LINEAR_MODEL.value:
            X_train_prepped = X_train.drop("location", axis=1)
            X_train_prepped = self.scaler.fit_transform(X_train_prepped)
            X_train_prepped = self.pca.fit_transform(X_train_prepped)
            self.model.fit(X_train_prepped, y_train)
        elif self.model_type.value == ModelType.LOCATION_LINEAR_MODEL.value:
            for loc in self.locations:
                location_mask = X_train["location"] == loc.value
                X_train_loc = X_train[location_mask]
                X_train_loc = X_train_loc.drop("location", axis=1)
                y_train_loc = y_train[location_mask]

                X_train_loc = self.scalers[loc].fit_transform(X_train_loc)
                X_train_loc = self.pcas[loc].fit_transform(X_train_loc)
                self.models[loc].fit(X_train_loc, y_train_loc)
