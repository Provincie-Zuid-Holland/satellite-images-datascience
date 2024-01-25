import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry.polygon import Polygon

from .utils import get_season_for_month


def get_flattened_pixels_for_polygon(
    dataset: rasterio.DatasetReader, polygon: Polygon
) -> pd.DataFrame:
    """
    Cuts polygon out of dataset and flattens the (6) bands of dataset into a single pandas DataFrame
    """
    cropped_to_polygon, _ = mask(dataset, [polygon], crop=True)

    df = pd.DataFrame(
        {
            "r": pd.Series(cropped_to_polygon[0].flatten(), dtype=float),
            "g": pd.Series(cropped_to_polygon[1].flatten(), dtype=float),
            "b": pd.Series(cropped_to_polygon[2].flatten(), dtype=float),
            "i": pd.Series(cropped_to_polygon[3].flatten(), dtype=float),
            "ndvi": pd.Series(cropped_to_polygon[4].flatten(), dtype=float),
            "height": pd.Series(cropped_to_polygon[5].flatten(), dtype=float),
        }
    )
    return df


def fill_pixel_columns(df: pd.DataFrame, label: str, image_name: str) -> pd.DataFrame:
    """
    Adds columns for the pixel dataframe.

    @param df: pixel dataframe
    @label: label given to the polygon these pixels belong to
    @image_name: filename of the tif file these pixels belong to
    @return pandas DataFrame, as df, but with additional columns
    """
    df["label"] = label
    df["image"] = image_name
    df["date"] = image_name[0:15]
    df["season"] = get_season_for_month(image_name[4:6])
    return df


def extract_dataframe_pixels_values_from_tif_and_polygons(
    tif_dataset: rasterio.DatasetReader,
    polygon_gdf: gpd.GeoDataFrame,
    name_tif_file: str,
) -> pd.DataFrame:
    """
    Filters polygons in polygon_gdf out of tif_dataset (for those polygons where row["name"] matches name_tif_file).
    Flattens the pixels in those polygons and adds several meta data columns

    @param tif_dataset: rasterio DatasetReader containing satellite imagery
    @param polygon_gdf: GeoDataFrame containing polygons in the tif_dataset area, labelled by the column 'Label'
    @name_tif_file: name of the tif_dataset object, so it can be matched with the correct row from polygon_df (polygon_gdf["name"])
    @return pandas DataFrame with a pixel per row
    """
    polygon_gdf = polygon_gdf

    dfs = []
    for _, row in polygon_gdf.iterrows():
        if row["name"] == name_tif_file:
            df_row = get_flattened_pixels_for_polygon(
                dataset=tif_dataset, polygon=row["geometry"]
            )
            df_row = fill_pixel_columns(df_row, row["Label"], image_name=name_tif_file)
            dfs += [df_row]
    if len(dfs) > 0:
        df = pd.concat(dfs)
        mask_non_empty_pixels = df["r"] != 0
        df = df[mask_non_empty_pixels].reset_index(drop=True)
    else:
        df = pd.DataFrame()

    return df
