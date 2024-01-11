import glob
import os
import re

import geopandas as gpd
import pandas as pd


def load_annotations_polygons(
    annotations_folder: str,
    annotations_polygon_filename_regex: str,
    location: str,
    images_folder: str,
) -> gpd.GeoDataFrame:
    """
    @param annotations_folder: folder conaintaining annotations file(s)
    @param annotations_polygon_filename_regex: filename of annotations geojson file

    @return GeopandasDataFrame containing the polygons of all annotations for the given location
    """
    dfs = []
    for filepath in glob.glob(
        os.path.join(annotations_folder, annotations_polygon_filename_regex)
    ):
        df = gpd.read_file(filepath)
        if df.crs != "epsg:28992":
            df = df.to_crs(epsg=28992)
        df.geometry = df.geometry.buffer(0)  # transform MultiPolygon to Polygon trick

        if not "name" in df.columns:
            image_date = re.findall("[0-9]{8}", filepath)
            image_file = glob.glob(f"{images_folder}/*{image_date}*{location}*tif")[0]
            image_filename_without_extension = os.path.split(image_file)[-1].split(
                ".tif"
            )[0]
            df["name"] = image_filename_without_extension

        dfs += [df]
    return pd.concat(dfs)
