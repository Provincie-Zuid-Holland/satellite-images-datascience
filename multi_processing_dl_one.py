import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob



if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32


    #path_to_tif_file = "E:/data/coepelduynen/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.tif"
    #out_path = "E:/output/Coepelduynen_segmentations/20211226_103526_SV1-01_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.shp"
    path_to_tif_file = "E:/data/coepelduynen/20210907_112017_SV1-04_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped.tif"
    out_path = "E:/output/Coepelduynen_segmentations/20210907_112017_SV1-04_SV_RD_11bit_RGBI_50cm_KatwijkAanZee_natura2000_coepelduynen_cropped_deep_learning.shp"
    tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    
    model = nso_ds_models.deep_learning_model(tif_kernel_generator)
    model.get_annotations(path_to_tif_file.split("/")[-1])
    model.set_standard_convolutional_network()
    model.train_model_on_sat_anno(path_to_tif_file.split("/")[-1])
        
    tif_kernel_generator.predict_all_output_multiprocessing(model, out_path , steps = 3)