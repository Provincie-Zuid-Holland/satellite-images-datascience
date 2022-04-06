import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob
import keras.models


if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32


    #path_to_tif_file = "E:/data/coepelduynen/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.tif"
    #out_path = "E:/output/Coepelduynen_segmentations/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.shp"
    path_to_tif_file = "E:/data/coepelduynen/20201231_105943_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped_ndvi_height.tif"
    out_path = "E:/output/Coepelduynen_segmentations/20201231_105943_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped_ndvi_height_deep_learning.shp"
    tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    
  
    model = keras.models.load_model('C:/repos/satellite-images-nso-datascience/standard_convolutional_network_2019_AWD_RGBI_NDVI_H')   
    tif_kernel_generator.predict_all_output_multiprocessing_keras(model, out_path , steps = 15, keras_break_size = 1000)