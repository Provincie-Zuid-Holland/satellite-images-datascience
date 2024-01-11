import os

import geopandas as gpd


def load_annotations_polygons(
    annotations_folder: str, annotations_polygon_filename: str
) -> gpd.GeoDataFrame:
    """
    @param annotations_folder: folder conaintaining annotations file(s)
    @param annotations_polygon_filename: filename of annotations geojson file

    @return GeopandasDataFrame containing the polygons of all annotations for the given location
    """
    annotations_polygons_gdf = gpd.read_file(
        os.path.join(annotations_folder, annotations_polygon_filename)
    )
    if annotations_polygons_gdf.crs != "epsg:28992":
        annotations_polygons_gdf = annotations_polygons_gdf.to_crs(epsg=28992)
    annotations_polygons_gdf.geometry = annotations_polygons_gdf.geometry.buffer(
        0
    )  # transform MultiPolygon to Polygon trick

    return annotations_polygons_gdf
