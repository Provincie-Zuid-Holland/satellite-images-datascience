import numpy as np
import pandas as pd

def euclidean_distance_kernels(kernel_x, kernel_y):
    return np.sum(np.abs(kernel_x - kernel_y))



class euclidean_distance_model:


    def __init__(self, a_kernel_generator):
           self.kernel_generator =  a_kernel_generator

    def set_ec_distance_annotations(self, path_annotations = "././annotations/coepelduynen_annotations.csv"):

        annotations = pd.read_csv(path_annotations)
        annotations['rd_x_y'] = annotations.apply(lambda x: self.kernel_generator.get_x_y(x['x_cor'], x['y_cor'] ),axis=1)
        annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.get_kernel_for_x_y(x['rd_x_y'][0],x['rd_x_y'][1]), axis=1)
        self.class_kernels = annotations
        
    def get_annotations(self):

        return self.class_kernels

    def predict_kernel(self, kernel):

        return self.class_kernels[self.class_kernels.index == self.class_kernels.apply(lambda x: euclidean_distance_kernels(x['kernel'], kernel), axis=1).idxmin()]['label'].values[0]