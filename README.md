# Introduction 

This repository contains all the models used in image object detection and image segmentation of the satellites images of the NSO used by the Province of South Holland.
Through out this document I will be referring to image object detection and image segmentation as segmentation.

# Model input.
 
The following figure represents the input we have for a model:

![Alt text](basic_model_input.png?raw=true "Title")


This inout data is generated in .tif files which is done in the other satellite images nso github repo [Here](https://github.com/Provincie-Zuid-Holland/satellite_images_nso/)

# (Image Processing) Kernels.

For the models (image processing) kernels are extracted from the original .tif satellite images.
Code is written here which should make this easy to do.

The added value this repo tries to contribute is to automatically segment all the pixels in a given satellite .tif file in the faster way possible. In this example we will use a euclidean distance model to segment all the pixels in a .tif file into segments that are specified in a annotations file.

See this code example:

```python

import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
path_to_tif_file = "<PATH_TO_NSO_SAT_IMG.TIF>"
out_path = "<PATH_TO_OUTPUT_FILE.shp"

x_size_kernel = 32
y_size_kernel = 32

# Extract the x row and y column.
x_row = 16
y_row = 4757

# Setup up a kernel generator for this .tif file.
tif_kernel_generator = nso_tif_kernel.nso_tif_kernel(path_to_tif_file, x_size_kernel, y_size_kernel)

kernel = tif_kernel_generator.get_kernel_for_x_y(x_row,y_row )
kernel.shape
#output: (4, 32, 32)
# This .tif file contains 4 dimensions in RGBI 

# Set a fade kernel which gives more weight to the centre pixel in the kernel.
tif_kernel_generator.set_fade_kernel()

# Make a euclidean distance model.
euclidean_distance_model = nso_ds_models.euclidean_distance_model(tif_kernel_generator)

# Set annotations for the model to predict on.
euclidean_distance_model.set_ec_distance_custom_annotations(path_to_tif_file.split("/")[-1], fade=True)

# Predict all the pixels in a .tif file with a particular model and stores the dissolved results in the out_path file.     
tif_kernel_generator.predict_all_output_multiprocessing(euclidean_distance_model, out_path , steps = 3, fade=True)
```



# Models.

Since the satellite files contains a lot of pixels and we predict per pixel, sometimes 3 billion, we have used different models in order to increase performance for larger satellite images.
## Euclidean distance Model.

This model simply calculates the Euclidean distance to pre annotated pixel kernels and then basis it's predictions on the shortest distance.

The following image demonstrates this model:

![Alt text](simple_model.png?raw=true "Title")

Predicting with Euclidean distance model is implement in the following way:

```python
import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
import nso_ds_classes.nso_ds_models as nso_ds_models


# Set up a kernel generator.
x_kernel_width = 32
y_kernel_height = 32


path_to_tif_file = "<PATH_TO_NSO_SAT_IMG.TIF>"
out_path = "<PATH_TO_OUTPUT_FILE.shp"


tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width, y_kernel_height)

# Setup a euclidean distance model based on the tif kernel generator.
euclidean_distance_model = nso_ds_models.euclidean_distance_model(tif_kernel_generator)

# Load the model with annotations wich have been presupplied in a .csv file.
euclidean_distance_model.set_ec_distance_annotations()

# Get a random kernel based on a row and a column position.
x_row = 50
y_row = 4757
kernel = tif_kernel_generator.get_kernel_for_x_y(x_row,y_row)

# Make a prediction based on this kernel.
euclidean_distance_model.predict_kernel(kernel)
#output: "bos"
```

## Deep learning Model.

We use Keras in python and made a deep learning with following architecture:

![Alt text](deep_learning_architecture.png?raw=true "Title")

Which is used in the following way:

```python
    import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
    import nso_ds_classes.nso_ds_models as nso_ds_models

    x_kernel_width = 32
    y_kernel_height = 32

    path_to_tif_file = "<PATH_TO_TIF_FILE>"
    out_path = "<PATH_TO_OUTPUT_FILE"
    tif_kernel_generator = nso_tif_kernel.nso_tif_kernel_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

    # Set a model on a .tif generator. 
    model = nso_ds_models.deep_learning_model(tif_kernel_generator)
    # Get annotations for a particular .tif file.
    model.get_annotations(path_to_tif_file.split("/")[-1])
    # Set a type of deep learning network.
    model.set_standard_convolutional_network()
    # Train the specificed model on the annotations.
    model.train_model_on_sat_anno(path_to_tif_file.split("/")[-1])

    # Use the model to predict all the pixels in a .tif in a multiprocessing way.    
    tif_kernel_generator.predict_all_output_multiprocessing(model, out_path , steps = 3)

```

# Author
Michael de Winter

Jeroen Esseveld.
# Contact

Contact us at vdwh@pzh.nl


