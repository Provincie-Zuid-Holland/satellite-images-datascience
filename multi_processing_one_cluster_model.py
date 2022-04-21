import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob
from nso_ds_classes.nso_ds_models import cluster_annotations_stats_model 
import nso_ds_classes.nso_ds_cluster as nso_ds_cluster 
from os.path import exists


if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32


    for file in glob.glob("E:/data/coepelduynen/*ndvi_height.tif"):

        path_to_tif_file = file.replace("\\","/")
        out_path = "E:/output/Coepelduynen_segmentations/"+path_to_tif_file.split("/")[-1].replace(".tif","_normalised_cluster_model.shp")
        tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

        cluster_centers_file = "./"+path_to_tif_file.split("/")[-1].split(".tif")[0]+"_normalised_cluster_centers.csv"
        
        if exists(cluster_centers_file) is False:

            a_nso_cluster_break = nso_ds_cluster.nso_cluster_break(tif_kernel_generator)
            a_nso_cluster_break.make_clusters_centers(cluster_centers_file)
        else:
            print("Previous cluster centers found")

        #cluster_centers_file = cluster_centers_file
        #a_cluster_annotations_stats_model = cluster_annotations_stats_model(cluster_centers_file)

            
       # tif_kernel_generator.predict_all_output_multiprocessing(a_cluster_annotations_stats_model, out_path , steps = 3, pixel_values=True)