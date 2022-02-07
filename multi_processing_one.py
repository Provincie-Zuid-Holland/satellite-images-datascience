import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob



if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32


    path_to_tif_file = "E:/data/coepelduynen/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.tif"
    out_path = "E:/output/Coepelduynen_segmentations/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.shp"
    tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    euclidean_distance_model = nso_ds_models.euclidean_distance_model(tif_kernel_generator)
    euclidean_distance_model.set_ec_distance_custom_annotations(path_to_tif_file.split("/")[-1])
    
    euclidean_distance_model.get_class_label(0)
        
    tif_kernel_generator.predict_all_output_multiprocessing(euclidean_distance_model, out_path , steps = 3)