import rasterio
import numpy as np
import pandas as pd
import glob
import os
import contextlib
import io
from satellite_images_nso_tif_model_iterator.tif_model_iterator import (
    tif_model_iterator,
)
from satellite_images_nso_tif_model_iterator.filenames.file_name_generator import (
    OutputFileNameGenerator,
)


class custom_model_metrics:
    """

    Custom model metrics for Remote Sensing of currently only waterplants and sandarea's.

    These metrics include difficult pixels and small test .tif files.

    @author: Michael de Winter.
    """

    def __init__(self, amodel, ascaler, alocation, asatellite_constellation):
        self.model = amodel
        self.scaler = ascaler
        self.location = alocation
        self.satellite_constellation = asatellite_constellation

        # Init model locations
        if alocation == "Voornes Duin":
            self.selected_features = [
                "r",
                "g",
                "b",
                "n",
                "e",
                "d",
                "ndvi",
                "re_ndvi",
                "height",
            ]

        elif alocation == "Coepelduynen":
            # Optimal parameters and features
            self.selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

        elif alocation == "Schippersgat" and asatellite_constellation == "Superview":
            # Optimal parameters and features
            self.selected_features = ["r", "g", "b", "i", "ndvi", "height"]

        elif alocation == "Schippersgat" and asatellite_constellation == "PNEO":
            # Optimal parameters and features
            self.selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

        elif alocation == "Nieuwkoopse_plassen" and asatellite_constellation == "PNEO":
            # Optimal parameters and features
            self.selected_features = ["r", "g", "b", "n", "e", "d", "ndvi", "re_ndvi"]

        elif (
            alocation == "Nieuwkoopse_plassen"
            and asatellite_constellation == "Superview"
        ):
            # Optimal parameters and features
            self.selected_features = ["r", "g", "b", "i", "ndvi", "ndwi"]

    def metric_difficult_pixels(self):

        fault_check = []

        # Fill with more locations than only nieuwkoopse plassen, when this project goes on.
        if (
            self.location == "Nieuwkoopse_plassen"
            and self.satellite_constellation == "Superview"
        ):

            fault_check.append(
                [
                    "Water",
                    self.model.predict(
                        self.scaler.transform([[275, 208, 117, 219, 97, 88]])
                    )
                    == "Water",
                ]
            )

            fault_check.append(
                [
                    "Water",
                    self.model.predict([[1024, 656, 575, 416, 122, 57]]) == "Water",
                ]
            )

        if len(fault_check) == 0:
            return (
                "Error no difficult pixels found for the location or the constellation."
            )

        return fault_check

    def metrics_on_small_tif_files(self):

        falses = 0

        if (
            self.satellite_constellation == "Superview"
            and self.location == "Nieuwkoopse_plassen"
        ):
            collect_files = glob.glob(
                os.path.abspath("./tests/test_input/").replace("\\", "/")
                + "/Nieuwkoopse_plassen_raster_images/*SV*.tif"
            )
        elif (
            self.satellite_constellation == "PNEO"
            and self.location == "Nieuwkoopse_plassen"
        ):
            collect_files = glob.glob(
                os.path.abspath("./tests/test_input/").replace("\\", "/")
                + "/Nieuwkoopse_plassen_raster_images/*PNEO*.tif"
            )

        return_metrics = []

        for afile in collect_files:
            afile = afile.replace("\\", "/")

            df = self.__raster_to_dataframe(afile)

            df["filename"] = afile
            df["date"] = afile.split("/")[-1][0:15]
            df["label"] = afile.split("_test")[0].split("_")[-1]

            predict_value = (
                pd.Series(
                    self.model.predict(
                        self.scaler.transform(df[self.selected_features])
                    )
                )
                .value_counts()
                .index[0]
            )

            return_metrics.append(
                [
                    afile,
                    predict_value == afile.split("_test")[0].split("_")[-1],
                    predict_value,
                    afile.split("_test")[0].split("_")[-1],
                ]
            )

        return return_metrics

    def metrics_on_small_tif_file_iterator(self):
        """
        TODO: Make a decision if this should stay here or go to the iterator repo, since it's almost the same
        as the last one on with the iterator package.
        """
        # Predict small .tif files.
        falses = -1

        for a_tif_file in glob.glob(
            os.path.abspath("./tests/test_input/").replace("\\", "/") + "*SV*.tif"
        ):
            a_tif_file = a_tif_file.replace("\\", "/")
            print(a_tif_file)

            with contextlib.redirect_stdout(io.StringIO()):
                output_file_name_generator = OutputFileNameGenerator(
                    output_path=os.path.abspath("./tests/test_output/"),
                    output_file_name=os.path.abspath("./tests/test_output/")
                    + a_tif_file.split("/")[-1].replace(".tif", ".parquet"),
                )

                nso_tif_kernel_iterator_generator = (
                    tif_model_iterator.TifModelIteratorGenerator(
                        path_to_tif_file=a_tif_file,
                        model=self.model,
                        output_file_name_generator=output_file_name_generator,
                        parts=2,
                        normalize_scaler=self.scaler,
                        column_names=self.selected_features,
                        dissolve_parts=False,
                        square_output=False,
                        skip_done_part=False,
                    )
                )

                nso_tif_kernel_iterator_generator.predict_all_output()

                if falses == -1:
                    falses = 0

                collect_files = glob.glob(
                    os.path.abspath("./tests/test_output/") + "*SV*.parquet"
                )
                for afile in collect_files:
                    afile = afile.replace("\\", "/")
                    print(afile)

                    print(pd.read_parquet(afile)["label"].value_counts())

                    print(afile.split("_test")[0].split("_")[-1])
                    if (
                        pd.read_parquet(afile)["label"].value_counts().index[0]
                        != afile.split("_test")[0].split("_")[-1]
                    ):
                        # TODO: Make the error ratings better.
                        print("Wrong!!!!!!!")
                        falses = falses + 1
                        os.remove(afile)

        false_accuracy = falses / len(glob.glob(collect_files))
        print("False rating off: " + str(falses / len(glob.glob(collect_files))))

        return false_accuracy

    def __raster_to_dataframe(self, a_tif_file) -> pd.DataFrame:
        """
        Read in a raster file and make a pandas dataframe from it.

        @param a_tif_file: path to a tif file from which a pandas dataframe will be made.
        @return a pandas dataframe based on the pixels from a tif file.
        """
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
            columns=self.selected_features + ["rd_x", "rd_y"],
        )

        return df[(df[["r", "g", "b"]] != 0).any(axis="columns")]

    def get_location(self):
        return self.location

    def get_satellite_constellation(self):
        return self.satellite_constellation
