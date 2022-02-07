import numpy as np
import pandas as pd
import geopandas as gpd

def euclidean_distance_kernels(kernel_x, kernel_y):
    return np.sum(np.abs(kernel_x - kernel_y))



class euclidean_distance_model:

    """
    A model that simply predicts a pixel kernel based on the euclidean distance between pre annotated kernels.
    
    """


    def __init__(self, a_kernel_generator):
        """
        Init a euclidean distance model based on a .tif kernel generator.

        @param a_kernel_generator:  a .tif generator, see nso tif kernel for what this is.
        """
        self.kernel_generator =  a_kernel_generator

    def set_ec_distance_baseline_annotations(self, path_annotations = "././annotations/coepelduynen_annotations.csv", fade = False):
        """
        Set annotations for thi euclidean distance model based on a .csv file.

        @param ath_annotations
        
        """

        annotations = pd.read_csv(path_annotations)
        annotations= annotations = annotations[annotations['date'] == "baseline"]
        annotations = pd.concat([annotations, annotations.apply(lambda x: self.kernel_generator.get_x_y(x['x_cor'], x['y_cor'] ),axis=1)], axis='columns')

        if fade == False:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y']), axis=1)
        elif fade == True:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.fade_tile_kernel(self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y'])), axis=1)

        self.class_kernels = annotations
    
    def set_ec_distance_custom_annotations(self, sat_name, path_annotations = "././annotations/coepelduynen_annotations.csv", fade = False):
        """ 
        Set custom predict kernels based on a sattelite name.
        """
        annotations = pd.read_csv(path_annotations) 
        annotations = annotations[annotations['date'] == sat_name]
        annotations[['wgs84_e', 'wgs84_n']] = annotations['WGS84'].dropna().str.split(",",expand=True,)
        annotations = gpd.GeoDataFrame(annotations, geometry=gpd.points_from_xy(annotations.wgs84_e,annotations.wgs84_n))
        annotations = annotations.set_crs(epsg=4326)
        annotations = annotations.to_crs(epsg=28992)

        annotations['rd_x'] = annotations['geometry'].x
        annotations['rd_y']  = annotations['geometry'].y

        annotations[["rd_x", "rd_y"]] = annotations.apply(lambda x: self.kernel_generator.get_x_y(x['rd_x'], x['rd_y'] ),axis=1)

        if fade == False:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y']), axis=1)
        elif fade == True:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.fadify_kernel(self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y'])), axis=1)

        self.class_kernels = annotations.reset_index()

    def set_custom_kernels(self, aclass_kernels):
        self.class_kernels = aclass_kernels   
        
    def get_annotations(self):

        return self.class_kernels

    def predict_class_name(self, kernel):
        """
        Predict the class of a kernel based on annotations.
        
        """
        return self.class_kernels[self.class_kernels.index == self.class_kernels.apply(lambda x: euclidean_distance_kernels(x['kernel'], kernel), axis=1).idxmin()]['label'].values[0]
    

    def predict(self,kernel):
        """
        Predict the class of a kernel based on annotations.

        @param kernel: A kernel to be predicted.
        @return: class in int type of the class.
        """
        return np.argmin([euclidean_distance_kernels(x,kernel)  for x in self.class_kernels['kernel'].values])

    def get_class_label(self,index):

        return self.class_kernels[self.class_kernels.index == int(index)]['label'].values[0]