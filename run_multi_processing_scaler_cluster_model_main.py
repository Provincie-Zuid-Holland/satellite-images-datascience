import nso_ds_classes.nso_tif_kernel_iterator as nso_tif_kernel_iterator
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob
from nso_ds_classes.nso_ds_models import cluster_scaler_BNDVIH_model
from nso_ds_classes.nso_ds_normalize_scaler import scaler_class_BNDVIH
import nso_ds_classes.nso_ds_cluster as nso_ds_cluster 
from os.path import exists
import run_settings

""" 
This is now the default model.

"""

if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 1
    y_kernel_height = 1


    for file in glob.glob(run_settings.input_folder_tif_files):

        path_to_tif_file = file.replace("\\","/")
        print(path_to_tif_file)
        out_path = run_settings.output_folder+path_to_tif_file.split("/")[-1].replace(".tif","_normalised_cluster_model.shp")
        tif_kernel_generator = nso_tif_kernel_iterator.nso_tif_kernel_iterator_generator(path_to_tif_file, x_kernel_width , y_kernel_height)
        cluster_centers_file = "./cluster_centers/normalized_5_BHNDVI_cluster_centers_dunes.csv"

               
        # Make clusters if a cluster file does not yet exist
        if exists(cluster_centers_file) is False:
            a_nso_cluster_break = nso_ds_cluster.nso_cluster_break(tif_kernel_generator)
            a_nso_cluster_break.get_stepped_pixel_df(output_name = path_to_tif_file.split("/")[-1], parts=2, begin_part=1, multiprocessing= True)
            
            a_nso_cluster_break.make_clusters_centers(cluster_centers_file)
        else:
            print("Previous cluster centers found")

        # Init the model to make prediction with.    
        a_cluster_annotations_stats_model = cluster_scaler_BNDVIH_model(cluster_centers_file)

        # This model needs scalers in order to be useful, check if they already exist.
        if exists("./scalers/"+path_to_tif_file.split("/")[-1]+"_band3.save") is False:
                print("No scalers found making scalers")
                a_nso_cluster_break = nso_ds_cluster.nso_cluster_break(tif_kernel_generator)
                a_nso_cluster_break.make_scaler_parts_pixel_df(output_name = path_to_tif_file.split("/")[-1], parts=2, specific_part=1, multiprocessing= True)

        # Initialize a scaler model.
        a_normalize_scaler_class_BNDVIH = scaler_class_BNDVIH( "./scalers/"+path_to_tif_file.split("/")[-1]+"_band3.save", \
                                                                            scaler_file_band5 = "./scalers/"+path_to_tif_file.split("/")[-1]+"_band5.save", \
                                                                            scaler_file_band6 = "./scalers/ahn4.save")



            
        tif_kernel_generator.predict_all_output(a_cluster_annotations_stats_model, out_path , parts = 3,  normalize_scaler= a_normalize_scaler_class_BNDVIH )