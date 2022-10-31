from typing import MutableMapping
import sklearn
import nso_ds_classes.nso_tif_kernel_iterator as nso_tif_kernel_iterator
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob
from os.path import exists
import run_settings
from nso_ds_classes.nso_ds_normalize_scaler import scaler_class_all
import pickle

"""

This class uses a sklearn model to predict all pixels in 

"""


if __name__ == '__main__':
    filename ="./models/randomforest_classifier_coepelduynen_contrast_annotations_grid_search_all_data_2019_2022_small.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    # Set a kernel generator.
    x_kernel_width = 1
    y_kernel_height = 1
    

    for file in glob.glob(run_settings.input_folder_tif_files):

            path_to_tif_file = file.replace("\\","/")
            print(path_to_tif_file)
            out_path = run_settings.output_folder+path_to_tif_file.split("/")[-1].replace(".tif","_rf_rgbindvih_model.shp")
            

            if int(path_to_tif_file.split("/")[-1][0:4]) <= 2019:
                ahn_type = "./scalers/ahn3.save"
            elif int(path_to_tif_file.split("/")[-1][0:4]) > 2019:
                ahn_type = "./scalers/ahn4.save"

            a_normalize_scaler_class_all = scaler_class_all(scaler_file_band1 = glob.glob("./scalers/"+path_to_tif_file.split("/")[-1]+"*band1*")[0].replace("\\","/"), \
                                                    scaler_file_band2 = glob.glob("./scalers/"+path_to_tif_file.split("/")[-1]+"*band2*")[0].replace("\\","/"), \
                                                    scaler_file_band3 = glob.glob("./scalers/"+path_to_tif_file.split("/")[-1]+"*band3*")[0].replace("\\","/"), \
                                                    scaler_file_band4 = glob.glob("./scalers/"+path_to_tif_file.split("/")[-1]+"*band4*")[0].replace("\\","/"), \
                                                    scaler_file_band5 = glob.glob("./scalers/"+path_to_tif_file.split("/")[-1]+"*band5*")[0].replace("\\","/"), \
                                                    scaler_file_band6 = ahn_type)

            # Initialize the iterator.
            tif_kernel_generator = nso_tif_kernel_iterator.nso_tif_kernel_iterator_generator(path_to_tif_file, x_kernel_width , y_kernel_height)
            tif_kernel_generator.predict_all_output(loaded_model, out_path , parts = run_settings.parts, normalize_scaler= a_normalize_scaler_class_all)