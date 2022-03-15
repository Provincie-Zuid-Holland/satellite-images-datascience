from locale import normalize
import numpy as np
import pandas as pd
import geopandas as gpd
from keras.models import Sequential
from keras import layers
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import false



class euclidean_distance_model:

    """
    A model that simply predicts a pixel kernel based on the euclidean distance between pre annotated kernels.
    
    """


    def __init__(self, a_kernel_generator,fade = False, normalize = False):
        """
        Init a euclidean distance model based on a .tif kernel generator.

        @param a_kernel_generator:  a .tif generator, see nso tif kernel for what this is.
        """
        self.kernel_generator =  a_kernel_generator
        self.fade = fade
        self.normalize = normalize

    def set_ec_distance_baseline_annotations(self, path_annotations = "././annotations/coepelduynen_annotations.csv"):
        """
        Set annotations for thi euclidean distance model based on a .csv file.

        @param ath_annotations
        
        """

        annotations = pd.read_csv(path_annotations)
        annotations= annotations = annotations[annotations['date'] == "baseline"]
        annotations = pd.concat([annotations, annotations.apply(lambda x: self.kernel_generator.get_x_y(x['x_cor'], x['y_cor'] ),axis=1)], axis='columns')

        if self.fade == False:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y']), axis=1)
        elif self.fade == True:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.fade_tile_kernel(self.kernel_generator.get_kernel_for_x_y(x['rd_x'],x['rd_y'])), axis=1)

        self.class_kernels = annotations
    
    def set_ec_distance_custom_annotations(self, sat_name = "empty" , path_annotations = "././annotations/coepelduynen_annotations.csv" ):
        """ 
        Set custom predict kernels based on a sattelite name.
        """

        if sat_name is "empty":
            sat_name = self.kernel_generator.get_sat_name()

        annotations = pd.read_csv(path_annotations) 
        annotations = annotations[annotations['date'] == sat_name]
        annotations[['wgs84_e', 'wgs84_n']] = annotations['WGS84'].dropna().str.split(",",expand=True,)
        annotations = gpd.GeoDataFrame(annotations, geometry=gpd.points_from_xy(annotations.wgs84_e,annotations.wgs84_n))
        annotations = annotations.set_crs(epsg=4326)
        annotations = annotations.to_crs(epsg=28992)

        annotations['rd_x'] = annotations['geometry'].x
        annotations['rd_y']  = annotations['geometry'].y

        annotations[["row_x", "column_y"]] = annotations.apply(lambda x: self.kernel_generator.get_x_y(x['rd_x'], x['rd_y'] ),axis=1)

        if self.fade == False and self.normalize == False:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.get_kernel_for_x_y(x["row_x"],x["column_y"]), axis=1)
        elif self.fade == True and self.normalize == False:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.fadify_kernel(self.kernel_generator.get_kernel_for_x_y(x["row_x"],x["column_y"])), axis=1)
        elif self.fade == False and self.normalize == True:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.normalize_tile_kernel(self.kernel_generator.get_kernel_for_x_y(x["row_x"],x["column_y"])), axis=1)
        elif self.fade == True and self.normalize == True:
            annotations['kernel'] = annotations.apply(lambda x: self.kernel_generator.fadify_kernel(self.kernel_generator.normalize_tile_kernel(self.kernel_generator.get_kernel_for_x_y(x["row_x"],x["column_y"]))), axis=1)

        self.class_kernels = annotations.reset_index()

    def set_custom_kernels(self, aclass_kernels):
        self.class_kernels = aclass_kernels   
        
    def get_annotations(self):

        return self.class_kernels
    
    def get_fade(self):
        return self.fade
    
    def get_normalize(self):
        return self.normalize

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


class generic_model:
    """
    
    A generic model for now.
    
    """


    def __init__(self, a_kernel_generator):
        """
        Init a euclidean distance model based on a .tif kernel generator.

        @param a_kernel_generator:  a .tif generator, see nso tif kernel for what this is.
        """
        self.kernel_generator =  a_kernel_generator

    def get_annotations(self, sat_name, path_annotations = "././annotations/coepelduynen_annotations.csv", fade = False):
            """ 
            Set custom predict kernels based on a sattelite name.

            @param sat_name: name of the satellite images.
            @param path_annotations: path to a .csv file for annotations.
            @param fade: wether to fade a kernel.
            @return annotations.
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

            return annotations.reset_index()
class deep_learning_model:


    def __init__(self, a_kernel_generator):
        self.kernel_generator = a_kernel_generator
        self.label_encoder = LabelEncoder()

    def set_standard_convolutional_network(self,size_x_matrix =32 ,size_y_matrix = 32,bands = 4, no_classes =5):
        model = standard_convolutional_network(size_x_matrix,size_y_matrix,bands, no_classes)
        model.compile(loss="sparse_categorical_crossentropy",
              optimizer= 'rmsprop',
              metrics=['accuracy'])
        self.model = model
    
    def train_model_on_sat_anno(self, sat_name, epochs = 32 ):
        self.get_annotations(sat_name) 
    

        y = self.label_encoder.fit_transform(self.annotations['label'].values)
        len_y = len(self.annotations)

        self.model.fit(np.concatenate(self.annotations["kernel"]).reshape(len_y,32,32,4).astype(int),y.reshape(len_y,1), epochs=32)


    def get_annotations(self, sat_name, path_annotations = "././annotations/coepelduynen_annotations.csv", fade = False):
            """ 
            Set custom predict kernels based on a sattelite name.

            @param sat_name: name of the satellite images.
            @param path_annotations: path to a .csv file for annotations.
            @param fade: wether to fade a kernel.
            @return annotations.
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

            self.annotations = annotations.reset_index()

    def predict(self, akernel):
        return self.label_encoder.inverse_transform([np.argmax(self.model.predict(np.concatenate(akernel).reshape(1,32,32,4).astype(int)))])[0]


### Deep learning models here.
def standard_convolutional_network(size_x_matrix =32 ,size_y_matrix = 32,bands = 4, no_classes =5):

     model = Sequential()
     model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size_x_matrix,size_y_matrix,bands)))
     model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
     model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
     model.add(Flatten())
     model.add(Dense(128, activation='relu'))
     model.add(Dense(no_classes, activation='softmax'))
     return model


def VGG_16_inspired_model(size_x_matrix =32 ,size_y_matrix = 32,bands = 4, outputs=5):
    model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(31,31, 7)))
    model.add(Convolution2D(28, 2, 2, activation='relu',input_shape=((bands, size_x_matrix, size_y_matrix))))
    model.add(Convolution2D(56, 2, 2, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputs, activation='softmax'))
    return model


# General functions here.
def euclidean_distance_kernels(kernel_x, kernel_y):
    return np.sum(np.abs(kernel_x - kernel_y))
