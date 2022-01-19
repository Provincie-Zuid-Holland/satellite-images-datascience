# Introduction 

This repository contains all the models used in image object dectection and image segmentation of the satellites images of the NSO used by the Provincie of South Holland.
Through out this document I will be referening to image object dectection and image segmentation as segmentation.


# Model input.

The following figure represents the input we have for a model:

![Alt text](basic_model_input.png?raw=true "Title")


# (Image Processing) Kernels.

For the models (image processing) kernels are extracted from the original .tif satellite images.
Code is written here which should make this easy to do.

See this code example:

```python


import nso_ds_classes.nso_tif_kernel as nso_tif_kernel
path_to_tif_file = "<PATH_TO_NSO_SAT_IMG.TIF>"

x_size_kernel = 32
y_size_kernel = 32

# Extract the x row and y column.
x_row = 16
y_row = 4757

# Setuo up a kernel generator for this .tif file.
tif_kernel_generator = nso_tif_kernel.nso_tif_kernel(path_to_tif_file, x_size_kernel, y_size_kernel)

kernel = tif_kernel_generator.get_kernel_for_x_y(x_row,y_row )
kernel.shape
#output: (4, 32, 32)
```
# Models.

Since the satellite files contains a lot of pixels, sometimes 3 billion, we have used different models in order to increase performance for larger satellite images.
## Euclidean distance Model.


This model simply calculates the euclidean distance to pre annotated pixel kernels and then basis it's predictions on the shortest distance.

The following image demonstrates this model:


![Alt text](simple_model.png?raw=true "Title")

## Deep learning.

We use keras in python with following architecture:

![Alt text](deep_learning_architecture.png?raw=true "Title")


# Author
Michael de Winter

Jeroen Esseveld.
# Contact

Contact us at vdwh@pzh.nl

