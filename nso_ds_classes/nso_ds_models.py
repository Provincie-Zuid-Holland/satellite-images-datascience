import numpy as np


def euclidean_distance_kernels(kernel_x, kernel_y):
    return np.sum(np.abs(kernel_x - kernel_y))


def eu_d_predict(kernel, class_kernels, class_name):
    return class_name[np.argmin([euclidean_distance_kernels(kernel,class_tile) for class_tile in class_kernels])]
