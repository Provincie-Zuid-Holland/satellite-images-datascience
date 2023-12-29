import os

import pandas as pd
import rasterio


class FlattenedRGBImageLoader:
    def __init__(self, filename: str, folder_path: str):
        self.filename = filename
        self.filepath = os.path.join(folder_path, self.filename)

    def load_flattened_rgb_values(self) -> pd.DataFrame:
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
