import nso_ds_classes.nso_tif_kernel_iterator as nso_tif_kernel_iterator
import nso_ds_classes.nso_ds_models as nso_ds_models
import nso_ds_classes.nso_tif_sampler as nso_tif_sampler
import nso_ds_classes.nso_ds_cluster as nso_ds_cluster
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np



x_kernel_width = 32
y_kernel_height = 32


path_to_tif_file = "E:/data/coepelduynen/20200625_112015_SV1-03_SV_RD_11bit_RGBI_50cm_Rijnsburg_natura2000_coepelduynen_cropped_ndvi_height.tif"
tif_kernel_generator = nso_tif_kernel_iterator.nso_tif_kernel_iterator_generator(path_to_tif_file, x_kernel_width , y_kernel_height)

a_nso_cluster_break = nso_ds_cluster.nso_cluster_break(tif_kernel_generator)
a_nso_cluster_break.retrieve_stepped_cluster_centers()