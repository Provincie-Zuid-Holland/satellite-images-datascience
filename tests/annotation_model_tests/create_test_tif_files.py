from rasterio.mask import mask
import rasterio
import glob
import geopandas as gpd

"""

This script creates test tif files based on pixel for model testing purposes.
Small tif files make data reading in a lot faster.


@author: Michael de Winter
"""


def crop_raster_based_on_gdf(agdf, input_raster_path, output_raster_path):

    area_to_crop = agdf["geometry"]

    with rasterio.open(input_raster_path) as src:

        out_image, out_transform = rasterio.mask.mask(
            src, area_to_crop, crop=True, filled=True
        )
        out_profile = src.profile

        out_profile.update(
            {
                "driver": "GTiff",
                "interleave": "band",
                "tiled": True,
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": out_image.dtype,
            }
        )

        with rasterio.open(output_raster_path, "w", **out_profile) as dest:
            dest.write(out_image)
            dest.close()


# Path to where the annotations are stored.
annotation_folder_path = "C:/repos/satellite-images-nso-datascience/data/annotations/Nieuwkoopse_Plassen/*.geojson"
# Path to where the .tif files are on which the annotations have been made
tif_files_path = "E:/data/nieuwkoopse_plassen/"
# Path to where the output will be stored.
test_output_dir = "E:/output/test/Nieuwkoopse_plassen/"

print(
    "Creating test .tif files of one pixel for Nieuwkoopse Plassen outputted in folder: "
    + test_output_dir
)

for afile in glob.glob(annotation_folder_path):
    afile = afile.replace("\\", "/")
    print(afile)

    waterplanten_annotations = gpd.read_file(afile)
    waterplanten_annotations = waterplanten_annotations.to_crs("EPSG:28992")
    if "label" in waterplanten_annotations.columns:
        waterplanten_annotations["Label"] = waterplanten_annotations["label"]

    tif_file = glob.glob(
        tif_files_path + afile.split("/")[-1].split("_")[0] + "*ndwi*"
    )[0].replace("\\", "/")

    crop_raster_based_on_gdf(
        waterplanten_annotations[waterplanten_annotations["Label"] == "Water"][0:1],
        tif_file,
        test_output_dir + tif_file.split("/")[-1].replace(".tif", "_Water_test.tif"),
    )
    if (
        len(
            waterplanten_annotations[waterplanten_annotations["Label"] == "Waterplants"]
        )
        > 0
    ):
        crop_raster_based_on_gdf(
            waterplanten_annotations[
                waterplanten_annotations["Label"] == "Waterplants"
            ][0:1],
            tif_file,
            test_output_dir
            + tif_file.split("/")[-1].replace(".tif", "_Waterplants_test.tif"),
        )
    crop_raster_based_on_gdf(
        waterplanten_annotations[waterplanten_annotations["Label"] == "Ground"][0:1],
        tif_file,
        test_output_dir + tif_file.split("/")[-1].replace(".tif", "_Ground_test.tif"),
    )
