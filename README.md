# Introduction

This repository houses all the training data and models utilized in the PZH Natura 2000 remote sensing project. The aim of this project is to employ machine learning techniques on land satellite imagery to monitor the distribution of various nature types within protected areas over time, in alignment with diverse climate/nature policies. Therefore, the models herein predict the nature type for each pixel in a land satellite image, which could include Grass, Forest, Sand, etc., based on learned annotations.

The annotated pixels are stored as pandas dataframes in the data/annotations folder and are publicly available for anyone to train their own model. In essence, we used pixel values from the land satellite images to annotate various nature types. Our objective is to predict the type of nature based on these pixel values.

For further information, please refer to the readme.md file in the data/annotation folder.

This repository includes three types of pixel-based models: a custom unsupervised spectral contrast model, a Keras deep learning network, and a random forest model trained on annotations. However, after a series of experiments, we found the random forest model trained on annotations to be the most effective and efficient for our project.

!Please note that we no longer support models other than the annotated supervised learning model.!

Given the computation-intensive nature of pixel-based prediction, due to the vast quantity of pixels, the performance of a model is a critical factor. For details on how we run/inference these models on satellite images, please visit this repository: [Here](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_tif_model_iterator) .

Thus this repository is exclusively used for model training.

# Installation

When working with 64x Windows and Anaconda for your python environment management execute the following terminal commands in order:

```sh
conda create -n satellite-images-nso-datascience python=3.12 -y
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

Decisions can be made to normalized/scaled the RGBIH values between 0 and 1.
This is done because of the unique RGBIH values a satellite image can have due to atmospheric influence thus normalization could theoretically reduce this influence.

In the notebook /scalers_make_run/run_make_scalers_normalize.ipynb normalize/scaling is done.

The resulting data from this can be found at:

Coepelduynen:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/coepelduynen/annotations_pixel_dataframes/annotaties_coepelduynen_to_pixel_scaled.csv

Voornes Duin:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/Voornes_Duin/annotations_pixel_dataframes/VoornesDuin_polyg2pixel_scaled_new.pkl

# Annotations Models.

After experimentation with different unsupervised models, we came to the conclusion that annotated supervised models are the best and used models.
In conclusion we first have to make annotations on satellite images and based on these annotations train a model.

Look in the annotations_models/train_random_forest_classifier_model.ipynb notebook to our results on training this model.


# Cloud Detection Models

The basis of these models are the cropped satellite images available on the `pzh-blob-satelliet` blob storage in folder `satellite-images-nso\SV_50cm`. The files used are listed in `cloud_recognition/satellite-images-clouds.csv`. To recreate a cloud detection model or create a new one, those files should be downloaded and placed in a folder.

After that one can follow the `cloud_recognition\cloud_recognition_train_model.ipynb` notebook to retrain a model or train a new one. The `cloud_recognition\cloud_recognition_apply_model.ipynb` notebook shows how to apply an existing model to a tif file.

# How to train a new model

1. Retrieve images using the [satellite_images_nso_extractor](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_extractor) repository.
2. Use QGis to set up annotations for each of the images. Make polygons for each area of a specific label and add the following columns, before exporting as a geojson:
   - 'name': this is the name of the tif file (extension excluded)
   - 'Label': this is the label of the selected polygon.
3. Next run the 'data/annotations/transform_polygon_annotations_to_pixels.ipynb' with the appropriate variables.
4. Finally run 'annotations_models/train_random_forest_classifier_model.ipynb' notebook with the appropriate variables. This notebook trains a scaler & model, resulting in:
   - A pickle file containing the cross-validation models, scalers & metrics
   - A pickle file containing the model & scaler trained on all annotated data
5. See the model results by running 'mlflow ui' in the annotations_models folder

# Author

Michael de Winter

Pieter Kouyzer

Jeroen Esseveld



# Contact

Contact us at vdwh@pzh.nl
