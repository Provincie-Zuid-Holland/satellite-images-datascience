import pandas as pd

from cloud_recognition.data_loaders import FlattenedRGBImageLoader
from cloud_recognition.data_preparation import DataPreprocessor
from cloud_recognition.model_training import Natura2000CloudDetectionModel


def detect_clouds(model: Natura2000CloudDetectionModel, filepath: str):
    location = None
    if model.locations:
        for loc in model.locations:
            if loc.value in filepath:
                location = loc.value

        if not location:
            Warning(
                f"Location {location} unknown in cloud detection model. f{filepath} skipped."
            )
            return None
    image_loader = FlattenedRGBImageLoader(filepath=filepath)
    data_preprocessor = DataPreprocessor()

    rgb_df = image_loader.get_rgb_df()
    features = data_preprocessor.transform(rgb_df)
    features["location"] = location
    feature_df = pd.DataFrame([features])
    clouds = model.predict(feature_df)
    return bool(clouds.iloc[0])
