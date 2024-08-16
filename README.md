# Introduction

This repository contains all the necessary resources—training data, models, and code—used in the PZH (Natura 2000) remote sensing project on NSO satellites. The goal of this project is to leverage machine learning techniques to analyze satellite imagery, enabling the monitoring of land distribution across various (nature) types within protected Natura 2000 areas over time. Our approach involves training models to classify individual pixels into specific (nature) classes based on their spectral characteristics.

Through our research, we have determined that supervised machine learning models yield the best results. However, these models require extensive training data, which necessitated the manual annotation of satellite image pixels.

The annotated pixels are organized in pandas DataFrames, which are available for download in the data/annotations folder. These annotations are publicly accessible, allowing others to use them for their own model training. Essentially, we've labeled pixel values in the satellite imagery to correspond with different land types.

The trained models can be downloaded via the links provided in the saved_models folder, as detailed in the readme.md file.

Due to the computational demands of pixel-based predictions—given the sheer volume of pixels involved—model performance is a critical consideration. For comprehensive details on how we conduct inference using these models on satellite images, please refer to the repository
[Here](https://github.com/Provincie-Zuid-Holland/satellite_images_nso_tif_model_iterator).

Thus, this repository is exclusively used for model training.
# Installation

When working with 64x Windows and Anaconda for your python environment management execute the following terminal commands in order, however when only using annotation models this is not used:

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

Decisions can be made to normalized/scaled the spectral values between 0 and 1 or standard Distribution of mean 0 and variance 1.
This is done because of the unique spectral values a satellite image can have due to atmospheric influence thus normalization could theoretically reduce this influence.

This scaling is done as well as model training done in the notebook /models/annotations_models/train_random_forest_classifier_model.ipynb


The resulting data from this can be found at:

Coepelduynen:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/coepelduynen/annotations_pixel_dataframes/annotaties_coepelduynen_to_pixel_scaled.csv

Voornes Duin:
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/Voornes_Duin/annotations_pixel_dataframes/VoornesDuin_polyg2pixel_scaled_new.pkl

# Saved Models.

See the saved_models folders for download links to these models.


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
