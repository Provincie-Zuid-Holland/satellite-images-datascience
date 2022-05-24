import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob



if __name__ == '__main__':

    # Set a kernel generator.
    x_kernel_width = 32
    y_kernel_height = 32

    # Old stuff
    #path_to_tif_file = "E:/data/coepelduynen/20190308_111644_SV1-01_50cm_RD_11bit_RGBI_Oegstgeest_natura2000_coepelduynen_cropped.tif"
    #tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    #euclidean_distance_model = nso_ds_models.euclidean_distance_model(tif_kernel_generator)
    #euclidean_distance_model.set_ec_distance_baseline_annotations()

    for file in glob.glob("E:/data/coepelduynen/202[0|1]0[3:9]*ndvi_height.tif"):
        print("-------")
        print(file.replace("\\","/"))
        file = file.replace("\\","/") 
        tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(file, x_kernel_width , y_kernel_height)

        tif_kernel_generator.set_fade_kernel(fade_power = 0.2)

        watlerleiding_duinen_ahn_ndvi_model = nso_ds_models.waterleiding_ahn_ndvi_model(tif_kernel_generator, annotations_np_array = "./median_annotation_rgbi_ndvi_rgbi.npy", fade=False)
        
        tif_kernel_generator.predict_all_output(watlerleiding_duinen_ahn_ndvi_model,"E:/output/Coepelduynen_segmentations/"+file.split("/")[-1].replace(".tif",".shp"), parts = 3, fade = False)