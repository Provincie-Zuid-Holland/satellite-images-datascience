import os

import pandas as pd
import rasterio


class FlattenedRGBImageLoader:
    def __init__(self, filename: str, folder_path: str):
        self.filename = filename
        self.filepath = os.path.join(folder_path, self.filename)

    def get_rgb_df(self):
        """
        Loads flattened RGB values from self.filename and filters out black spots.
        """
        rgb_df = self._load_flattened_rgb_values()
        return self._filter_black_spots(rgb_df)

    def _load_flattened_rgb_values(self) -> pd.DataFrame:
        """
        Flattens all pixels from self.filename into a single array and returns the Red, Green and Blue bands into a pandas dataframe.
        """
        with rasterio.open(self.filepath) as dataset:
            rgb_df = pd.DataFrame(
                {
                    "red": dataset.read(1).flatten(),
                    "green": dataset.read(2).flatten(),
                    "blue": dataset.read(3).flatten(),
                }
            )
        return rgb_df

    @staticmethod
    def _filter_black_spots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters pixels that have 0's for Red, Green and Blue values
        """
        red_zero_mask = df["red"] != 0
        green_zero_mask = df["green"] != 0
        blue_zero_mask = df["blue"] != 0
        return df[red_zero_mask & green_zero_mask & blue_zero_mask]
