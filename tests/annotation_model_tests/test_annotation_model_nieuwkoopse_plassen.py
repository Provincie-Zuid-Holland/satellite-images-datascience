import pickle
import glob
import rasterio
import numpy as np
import pandas as pd
import contextlib
from tif_model_iterator import tif_kernel_iterator
from filenames.file_name_generator import OutputFileNameGenerator
import io
import sys

# Adjust this for the tests!!
model_path = "C:/repos/satellite-images-nso-datascience/saved_models/Superview_Nieuwkoopse_plassen_20190302_113613_to_20221012_104900_random_forest_classifier.sav"
location = "Nieuwkoopse_plassen"
satellite_constellation = "Superview"

# Put in settings file
test_tif_files_dir = "E:/output/test/Nieuwkoopse_plassen/*SV*.tif"


def raster_to_dataframe(a_tif_file):
    src = rasterio.open(a_tif_file)
    data = src.read()
    z_shape = data.shape[0]
    x_shape = data.shape[1]
    y_shape = data.shape[2]

    x_coordinates = [
        [x for y in range(0, data.shape[2])] for x in range(0, data.shape[1])
    ]
    y_coordinates = [
        [y for y in range(0, data.shape[2])] for x in range(0, data.shape[1])
    ]

    rd_x, rd_y = rasterio.transform.xy(src.transform, x_coordinates, y_coordinates)

    data = np.append(data, rd_x).reshape([z_shape + 1, x_shape, y_shape])
    data = np.append(data, rd_y).reshape([z_shape + 2, x_shape, y_shape])

    data = data.reshape(-1, x_shape * y_shape).transpose()

    df = pd.DataFrame(
        data,
        columns=selected_features + ["rd_x", "rd_y"],
    )

    return df[(df[["r", "g", "b"]] != 0).any(axis="columns")]


# Init model locations
if location == "Voornes Duin":
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi", "height"]


elif location == "Coepelduynen":
    # Optimal parameters and features
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]


elif location == "Schippersgat" and satellite_constellation == "Superview":
    # Optimal parameters and features
    selected_features = ["r", "g", "b", "i", "ndvi", "height"]


elif location == "Schippersgat" and satellite_constellation == "PNEO":
    # Optimal parameters and features
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]


elif location == "Nieuwkoopse_plassen" and satellite_constellation == "PNEO":
    # Optimal parameters and features
    selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]


elif location == "Nieuwkoopse_plassen" and satellite_constellation == "Superview":
    # Optimal parameters and features
    selected_features = ["r", "g", "b", "i", "ndvi", "ndwi"]


final_artefact = pickle.load(open(model_path, "rb"))


def test_difficult_pixels():

    if location == "Nieuwkoops_plassen" and satellite_constellation == "Superview":
        assert (
            final_artefact["model"].predict(
                final_artefact["scaler"].transform([[275, 208, 117, 219, 97, 88]])
            )
            == "Water"
        )

        assert (
            final_artefact["model"].predict([[1024, 656, 575, 416, 122, 57]]) == "Water"
        )


def test_predict_all_function():

    # Predict small .tif files.
    for a_tif_file in glob.glob(test_tif_files_dir):
        a_tif_file = a_tif_file.replace("\\", "/")
        print(a_tif_file)

        with contextlib.redirect_stdout(io.StringIO()):
            output_file_name_generator = OutputFileNameGenerator(
                output_path="E:/output/test/Nieuwkoopse_plassen/",
                output_file_name="E:/output/test/Nieuwkoopse_plassen/"
                + a_tif_file.split("/")[-1].replace(".tif", ".parquet"),
            )

            nso_tif_kernel_iterator_generator = (
                tif_kernel_iterator.TifKernelIteratorGenerator(
                    path_to_tif_file=a_tif_file,
                    model=final_artefact["model"],
                    output_file_name_generator=output_file_name_generator,
                    parts=1,
                    normalize_scaler=final_artefact["scaler"],
                    column_names=selected_features,
                    dissolve_parts=False,
                    square_output=False,
                    skip_done_part=False,
                )
            )

            nso_tif_kernel_iterator_generator.predict_all_output()

            falses = 0
            for afile in glob.glob("E:/output/test/Nieuwkoopse_plassen/*SV*.parquet"):
                afile = afile.replace("\\", "/")
                print(afile)

                print(pd.read_parquet(afile)["label"].value_counts())

                print(afile.split("_test")[0].split("_")[-1])
                if (
                    pd.read_parquet(afile)["label"].value_counts().index[0]
                    != afile.split("_test")[0].split("_")[-1]
                ):
                    print("Wrong!!!!!!!")
                    falses = falses + 1

            print(
                "False rating off: "
                + str(
                    falses
                    / len(glob.glob("E:/output/test/Nieuwkoopse_plassen/*SV*.parquet"))
                )
            )

            assert falses == 0


def test_directly_on_tif_files():

    falses = 0
    for afile in glob.glob("E:/output/test/Nieuwkoopse_plassen/*SV*.tif"):
        afile = afile.replace("\\", "/")
        print(afile)

        df = raster_to_dataframe(afile)
        print(df)

        df["filename"] = afile
        df["date"] = afile.split("/")[-1][0:15]
        df["label"] = afile.split("_test")[0].split("_")[-1]

        print(afile.split("_test")[0].split("_")[-1])
        print(
            pd.Series(
                final_artefact["model"].predict(
                    final_artefact["scaler"].transform(df[selected_features])
                )
            )
            .value_counts()
            .index[0]
        )
        if (
            pd.Series(
                final_artefact["model"].predict(
                    final_artefact["scaler"].transform(df[selected_features])
                )
            )
            .value_counts()
            .index[0]
            != afile.split("_test")[0].split("_")[-1]
        ):
            print("Wrong!!!!!!!")
            falses = falses + 1

    print(
        "False rating off: "
        + str(falses / len(glob.glob("E:/output/test/Nieuwkoopse_plassen/*SV*.tif")))
    )
    assert falses == 0
