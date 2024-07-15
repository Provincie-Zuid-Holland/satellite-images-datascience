from .data_loaders import FlattenedRGBImageLoader


def plot_hist(filename: str, folder_path: str):
    image_loader = FlattenedRGBImageLoader(
        filename=filename,
        folder_path=folder_path,
    )

    rgb_df = image_loader.get_rgb_df()
    rgb_df.hist(bins=100)
