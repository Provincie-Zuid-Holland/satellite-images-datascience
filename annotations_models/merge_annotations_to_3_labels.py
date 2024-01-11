import os

import geopandas as gpd


def convert_label(label: str) -> str:
    conversion = {
        "zand": "Zand",
        "water": "Water",
        "struweel": "Begroeid",
        "duinvallei": "Begroeid",
        "laag gras": "Begroeid",
        "laag vegetatie": "Begroeid",
        "bos": "Begroeid",
        "duin vallei": "Begroeid",
        "struwee": "Begroeid",
        "laag vegatatie": "Begroeid",
        "lssg gras": "Begroeid",
        "Asftal": "Asfalt",
    }
    if label in conversion.keys():
        return conversion[label]
    else:
        return label


# Set variables
folder_path = "../../Data/remote-sensing/annotations"
filename = "annotaties_VoornesDuin_gecorrigeerd_100124.geojson"


# Convert to correct filenames
original_filepath = os.path.join(folder_path, filename)
file_base, file_extension = filename.split(".")
converted_filename = f"{file_base}_3_labels.{file_extension}"
converted_filepath = os.path.join(folder_path, converted_filename)


annotations_gdf = gpd.read_file(original_filepath)

annotations_gdf["name"] = annotations_gdf["name"] + "_asphalt_crop"

# Display original Labels in gdf
print(annotations_gdf["Label"].unique())
annotations_gdf["Label"] = annotations_gdf["Label"].apply(convert_label)

no_asfalt_mask = annotations_gdf["Label"] != "Asfalt"
no_shadow_mask = annotations_gdf["Label"] != "Schaduw"
annotations_gdf = annotations_gdf[no_asfalt_mask & no_shadow_mask]

# Display converted Labels in gdf
print(annotations_gdf["Label"].unique())

annotations_gdf.to_file(converted_filepath, driver="GeoJSON", name=file_base)
