import glob
import os
import re

import geopandas as gpd
import pandas as pd

import fiona


def load_annotations_polygons_gpkg(path_to_gpkg: str) -> gpd.GeoDataFrame:
    gdfs = []
    for alayer in fiona.listlayers(path_to_gpkg):
        gdf = gpd.read_file(path_to_gpkg, layer=alayer)
        if gdf.crs != "epsg:28992":
            gdf = gdf.to_crs(epsg=28992)
        gdf["name"] = alayer

        gdfs += [gdf]
    return pd.concat(gdfs)


def load_annotations_polygons(
    annotations_folder: str,
    annotations_polygon_filename_regex: str,
    images_folder: str = None,
    image_regex: str = None,
) -> gpd.GeoDataFrame:
    """
    @param annotations_folder: folder conaintaining annotations file(s)
    @param annotations_polygon_filename_regex: filename of annotations geojson file
    @images_folder: folder containing tif files (only necessary if polygon files do not contain name column)
    @image_regex: regex to find correct if files in images_folder (only necessary if polygon files do not contain name column)

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
            image_file = glob.glob(f"{images_folder}/*{image_date}{image_regex}")[0]
            image_filename_without_extension = os.path.split(image_file)[-1].split(
                ".tif"
            )[0]
            df["name"] = image_filename_without_extension

        dfs += [df]
    return pd.concat(dfs)
