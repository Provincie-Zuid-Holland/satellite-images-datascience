## Overview

The .gpkg files are the hand drawn annotations files but these are only coordinates.
Red, Green, Blue, Infrared, Height (RGBIH) values still have to be extracted from the satellite .tif files images.

But premade data downlinks are given here which contain .csv files based on the hand drawn annotated satellite images pixels and there corresponding RGBI values.
There are 2 versions:

1. The raw RGB values, mind that these satellite images uses RGB values higher than 255.

2. And one with rescaled values between 0 en 1 based on a scaler trained on the specific satellite image from which the specific pixels are drawn from.
These scalers are done to hopefully normalize the influence of the atmosphere

We share these files so hopefully other organizations can from this as well.

### Download links annotated Red, Green, Blue, Infrared, Height (RGBIH) data

#### Coepelduynen 

https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/coepelduynen/annotations_pixel_dataframes/annotaties_coepelduynen_to_pixel.csv
https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/coepelduynen/annotations_pixel_dataframes/annotaties_coepelduynen_to_pixel_scaled.csv

### Voornes duinen

https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/Voornes_Duin/annotations_pixel_dataframes/VoornesDuin_polyg2pixel_new.pkl

https://e34a505986aa74678a5a0e0f.blob.core.windows.net/satellite-images-nso/Voornes_Duin/annotations_pixel_dataframes/VoornesDuin_polyg2pixel_scaled_new.pkl
