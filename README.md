# Introduction

This repository contains all the annotation training data and models used by the PZH Natura 2000 remote sensing project.
In short for this project we want to use machine learning on land satellite images to monitor each pixel distribution of various nature types in nature protected areas across time for various climate/nature policies.
So the models here predicts for each pixel in a land satellite image a nature type, these types could include: Grass,Forest,Sand ..etc based on the learned annotations.

Annotated pixels pandas dataframe are stored in the data/annotations folder, as such it is open for the public to train their own model.
This means that we took pixel values from in our case the SuperView land satellite images for annotated nature types.
Based on these pixel value we want to predict what kind of nature type it is.

Look in the data/annotation readme.md file for more information.

3 types of pixel based models are here: a custom unsupervised supervised spectral contrast model, a keras deep learning network and random forest model trained on annotations.

After some experimentation for our project we went for the random forest model trained on annotations, giving us the best model results for the best execution performance.

Since pixel based prediction is computationally intensive because of the large amount of pixels performance of a model is also a important criteria.
See this repository for how we run these models: [Here](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_tif_model_iterator)

# Installation

When working with 64x Windows and Anaconda for your python environment management execute the following terminal commands in order:

```sh
conda create -n satellite-images-nso-datascience python=3.9 -y
conda activate satellite-images-nso-datascience
pip install -r requirements.txt
pip install -e .
```

# Model input.

The following figure represents the input we have for a model:

![Alt text](basic_model_input.png?raw=true "Title")

This input data is generated in .tif files which is done in the other satellite images nso github repository [Here](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_extractor)

And for the height data here: [Here](https://github.com/Provincie-Zuid-Holland/vdwh_ahn_processing)

## Input normalization/scaling.

For all the models we have used we have normalized/scaled the RGBIH values between 0 and 1.
This is done because of the unique RGBIH values a satellite image can have due to atmospheric influence thus normalization could theoretically reduce this influence.

In the notebook /scalers_make_run/run_make_scalers_normalize.ipynb normalize/scaling is done.

The resulting data from this can be found at:

Coepelduynen:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/coepelduynen/annotations_pixel_dataframes/annotaties_coepelduynen_to_pixel_scaled.csv

Voornes Duin:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/Voornes_Duin/annotations_pixel_dataframes/VoornesDuin_polyg2pixel_scaled_new.pkl

# Running a model on (Image Processing) Kernels.

The main functionality of [this](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_tif_model_iterator) repository is to extract image kernels and the multiprocessing for loop for looping over all the pixels and/or image kernels in a given satellite .tif file to make predictions on them.

The following picture gives a illustration about how this extracting of kernels is done:
![Alt text](kernel_extract.png?raw=true "Title")

Here below we will have a code example about how this work. In this example we will use a Euclidean distance model to segment all the pixels in a .tif file into segments that are specified in a annotations file.

```python

import nso_ds_classes.nso_tif_kernel_iterator as nso_tif_kernel_iterator
import nso_ds_classes.nso_ds_models as nso_ds_models
path_to_tif_file = "<PATH_TO_NSO_SAT_IMG.TIF>"
out_path = "<PATH_TO_OUTPUT_FILE.shp"

# The kernel size will be 32 by 32
x_size_kernel = 32
y_size_kernel = 32

# Extract the x row and y column.
x_row = 16
y_row = 4757

# Setup up a kernel generator for this .tif file.
tif_kernel_generator = nso_tif_kernel_iterator. .nso_tif_kernel_iterator_generator(path_to_tif_file, x_size_kernel, y_size_kernel)

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

# Iterates and predicts all the pixels in a .tif file with a particular model and stores the dissolved results in the out_path file in a multiprocessing way. So this has to be run from a terminal.
tif_kernel_generator.predict_all_output(euclidean_distance_model, out_path , parts = 3, fade=True)
```

Note that this is functionality is also found at [this](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_extractor) repository.

# Models

[comment]: <> ( Since the satellite files contain a lot of pixels or extracted kernels and since we predict per pixel or extracted kernel, sometimes 3 billion pixels in total, we have used different models in order to increase performance for larger satellite images.)

We have tried different models all of which below are given descriptions, our custom Scaled Spectral Profile Contrast model, a random forest model based on annotations training data and a deep learning model based on annotations training data.

After some research we decided to go for the random forest model based on annotations based on annotations training data.
This model gave us the best execution performance and model performance, since we are predicting a lot of pixels, execution performance is also important

## Random forest model based on annotated training data.

After various testing we decided that annotating SuperView Satellite data is necessary to have objective dataset to score models on and the train a model on.
For which we handmade annotated Satellite data.

This ended being the used model based on it's accuracy and prediction speed performance.
The notebook where the model is trained can be found in ./annotations_models/Coepelduynen/random_forest/make_train_model_on_annotations_coepelduynen.ipynb

### Annotations.

We handmade all annotations by using Superview Satellite and drawing the different labels on them.
These labels vary by nature area, for the Coepelduynen we annotated the these labels.

| Labels              |
| ------------------- |
| Gras                |
| Zand                |
| Struweel            |
| Bos                 |
| Asfalt              |
| Schaduw             |
| Vochtige duinvallei |

Are found in the data/annotations folder, read the readme.md file for more information.

## Scaled Spectral Profile Contrast Model

This models can be seen as a combination of unsupervised and supervised models because it's require training annotation data which the other models do require.
Unsupervised because we first make Spectral Profile Contrast classes based on the descriptive medians of nature type data.

Based on these Spectral Profile Contrast classes we predict if a pixel is a certain nature type class.

Most of these models are stored in the nso_ds_classes /nso_ds_models.py file.

### Predetermined Spectral Profile Contrast classes

First Predetermined Spectral Profile Contrast classes centers are made which classify classes based on there contrasts. These classes were made with domain expertise and descriptive median data of nature types

These are the current cluster centers we are using:

![Alt text](cluster_centers_contrast_model.PNG?raw=true "Title")

Were band 3 is the color blue, band 5 is NDVI more about what NDVI is [Here](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/normalized-difference-vegetation-index#:~:text=The%20NDVI%20is%20a%20dimensionless,From%3A%20Environmental%20Research%2C%202018) and band 6 Lidar height data more information about which Lidar used [Here](https://www.ahn.nl)

These values are already scaled more about this in the next section.

### Value scaler

This model first calculates contrast for every different input values using scaler values between 0 en 1. This has to be done for each satellite image and for each input/band.
We use sklearn to make these min max scalers which have to be stored in a .save file which in turn is stored in the scalers folder.

For more about how min max scalers, read the sklearn page [Here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

### Input filtering

For the input we discovered that the atmosphere had a collinearity effect on the three red, green and blue values. As such we only pick the blue band in order to reduce the relative color weight in the prediction compared to NDVI and height data.
A other advantage of using the color blue is that it’s the best color to detect sand with. Detecting sand is particularly interesting for ecologists.

The following illustration exemplifies the eventual and final input we have for the contrast model:

![Alt text](contrast_model_input.png?raw=true "Title")

### Predicting

The following picture gives a easy illustration about how the prediction works:

![Alt text](Contrast_model_example.png?raw=true "Title")

For a particular pixel or kernel, in which blue, NDVI and height is used and scaled between 0 en 1, Euclidean distance is calculated for blue, NDVI and height to each cluster center. The shortest Euclidean distance to a class, is then the predicted class.

This can easily be interpreted as if for a given certain pixel or kernel the scaled blue color is between 1 and 0.7 this would mean that for this certain pixel/kernel sand for the input value blue. The same goes for the other input values NDVI and Height. The combinations of different input values gives the final predicted class.

### Example code

Below here we will give a code example of how to use this contrast model and added comments.

```python

from tif_model_iterator import tif_kernel_iterator as nso_tif_kernel_iterator
import nso_ds_classes.nso_ds_models as nso_ds_models
import glob
from nso_ds_classes.nso_ds_models import cluster_scaler_BNDVIH_model
from nso_ds_classes.nso_ds_normalize_scaler import scaler_class_BNDVIH
import nso_ds_classes.nso_ds_cluster as nso_ds_cluster
from os.path import exists

"""
This is now the default model.

"""

if __name__ == '__main__':

    # Set a kernel generator, in this case we will only use a kernel size of 1.
    x_kernel_width = 1
    y_kernel_height = 1

    # Loop through a directory which contains .tif files to be predicted.
    for file in glob.glob("<PATH>/<TO>/<TIF_DIRECTORY>/*blue_ndvi_height.tif"):

        # Setup a tif kernel iterator.
        path_to_tif_file = file.replace("\\","/")
        print(path_to_tif_file)
        out_path = "E:/output/Coepelduynen_segmentations/"+path_to_tif_file.split("/")[-1].replace(".tif","_normalised_cluster_model.shp")
        tif_kernel_generator = nso_tif_kernel_iterator.nso_tif_kernel_iterator_generator(path_to_tif_file, x_kernel_width , y_kernel_height)


        # Path to the predetermined classes.
        cluster_centers_file = "./cluster_centers/normalized_5_BHNDVI_cluster_centers.csv"


        # Load the class cluster centers into a euclidean distance model.
        a_cluster_annotations_stats_model = cluster_scaler_BNDVIH_model(cluster_centers_file)

        # This model needs scalers in order to be useful for a specific .tif file, check if they already exist. Make them if they don't already exist.
        if exists("./scalers/"+path_to_tif_file.split("/")[-1]+"_band3.save") is False:
                print("No scalers found making scalers")
                a_nso_cluster_break = nso_ds_cluster.nso_cluster_break(tif_kernel_generator)
                a_nso_cluster_break.make_scaler_parts_pixel_df(output_name = path_to_tif_file.split("/")[-1], parts=2, begin_part=1, multiprocessing= True)

        # Initialize a scaler model for the iterater to be used.
        a_normalize_scaler_class_BNDVIH = scaler_class_BNDVIH( "./scalers/"+path_to_tif_file.split("/")[-1]+"_band3.save", \
                                                                            scaler_file_band5 = "./scalers/"+path_to_tif_file.split("/")[-1]+"_band5.save", \
                                                                            scaler_file_band6 = "./scalers/ahn4.save")

        # The main iterator loop, if no multiprocessing variable is giving, it automatically will do multiprocessing and thus has to be run from a kernel.
        tif_kernel_generator.predict_all_output(a_cluster_annotations_stats_model, out_path , parts = 3,  normalize_scaler= a_normalize_scaler_class_BNDVIH )

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
    tif_kernel_generator.predict_all_output(model, out_path , parts = 3)

```

# Cloud Detection Models

The basis of these models are the cropped satellite images available on the `pzh-blob-satelliet` blob storage in folder `satellite-images-nso\SV_50cm`. The files used are listed in `cloud_recognition/satellite-images-clouds.csv`. To recreate a cloud detection model or create a new one, those files should be downloaded and placed in a folder.

After that one can follow the `cloud_recognition\cloud_recognition_train_model.ipynb` notebook to retrain a model or train a new one. The `cloud_recognition\cloud_recognition_apply_model.ipynb` notebook shows how to apply an existing model to a tif file.

# Author

Michael de Winter

Jeroen Esseveld.

Pieter Kouyzer

# Contact

Contact us at vdwh@pzh.nl
