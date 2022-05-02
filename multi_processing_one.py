import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob



if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32


    #path_to_tif_file = "E:/data/coepelduynen/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.tif"
    #out_path = "E:/output/Coepelduynen_segmentations/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.shp"
    path_to_tif_file = "E:/data/coepelduynen/20200625_112015_SV1-03_SV_RD_11bit_RGBI_50cm_Rijnsburg_natura2000_coepelduynen_cropped_ndvi_height.tif"
    out_path = "E:/output/Coepelduynen_segmentations/20200625_112015_SV1-03_SV_RD_11bit_RGBI_50cm_Rijnsburg_natura2000_coepelduynen_cropped_ndvi_height_ec_distance.shp"
    tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    tif_kernel_generator.set_fade_kernel(bands=6)

    euclidean_distance_model = nso_ds_models.euclidean_distance_model(tif_kernel_generator, normalize=True)
    euclidean_distance_model.set_ec_distance_custom_annotations()
    

        
    tif_kernel_generator.predict_all_output_multiprocessing(euclidean_distance_model, out_path , parts = 3)