import nso_ds_classes.nso_tif_kernel_iterator as nso_tif_kernel_iterator
import glob
import run_settings
from nso_ds_classes.nso_ds_normalize_scaler import scaler_normalizer_retriever

if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 1
    y_kernel_height = 1


    for file in glob.glob(run_settings.input_folder_tif_files):

        path_to_tif_file = file.replace("\\","/")
        tif_kernel_generator = nso_tif_kernel_iterator.nso_tif_kernel_iterator_generator(path_to_tif_file, x_kernel_width , y_kernel_height)
        

        a_scaler_normalizer_retriever = scaler_normalizer_retriever(tif_kernel_generator)
        a_scaler_normalizer_retriever.make_scalers_pixel_df(output_name = path_to_tif_file.split("/")[-1], parts=2, specific_part=1, multiprocessing= True)
        