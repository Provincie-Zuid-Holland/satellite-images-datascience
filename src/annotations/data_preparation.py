import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

from .utils import get_season_for_month


def extract_dataframe_pixels_values_from_tif_and_polygons(
    path_to_tif: str, path_to_polygons: str
):
    # TODO: Move reading of files outside of this function
    # TODO: Move row wise function into a separate function
    # TODO: make r==0 filter more explicit
    geo_file = gpd.read_file(path_to_polygons)
    # TODO: Remove filter below
    geo_file = geo_file[:10]
    src = rasterio.open(path_to_tif)
    dfs = []
    name_tif = path_to_tif.split("/")[-1].split(".")[0]
    print(name_tif)

    if geo_file.crs != "epsg:28992":
        geo_file = geo_file.to_crs(epsg=28992)

    for _, row in geo_file.iterrows():
        if row["name"] == name_tif:
            out_image, _ = mask(src, row["geometry"], crop=True)

            df_row = pd.DataFrame(
                {
                    "r": pd.Series(out_image[0].flatten(), dtype=float),
                    "g": pd.Series(out_image[1].flatten(), dtype=float),
                    "b": pd.Series(out_image[2].flatten(), dtype=float),
                    "i": pd.Series(out_image[3].flatten(), dtype=float),
                    "ndvi": pd.Series(out_image[4].flatten(), dtype=float),
                    "height": pd.Series(out_image[5].flatten(), dtype=float),
                }
            )
            df_row["label"] = row["Label"]
            df_row["image"] = path_to_tif.split("/")[-1]
            df_row["date"] = path_to_tif.split("/")[-1][0:15]
            df_row["season"] = get_season_for_month(path_to_tif.split("/")[-1][4:6])
            dfs += [df_row]
    src.close()
    if len(dfs) > 0:
        df = pd.concat(dfs)
        df = df[df["r"] != 0].reset_index(drop=True)
        print(len(df))
    else:
        df = pd.DataFrame()

    return df
