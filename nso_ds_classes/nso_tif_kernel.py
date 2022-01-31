from xml.dom import ValidationErr
import numpy as np 
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import pandas as pd
import nso_ds_classes.nso_ds_output as nso_ds_output 
from tqdm import tqdm
import os
import glob
import geopandas as gpd
from multiprocessing import Pool
import itertools
from timeit import default_timer as timer

"""
    This code is used to extract image processing kernels from nso satellite images.

    For more information what kernels are: https://en.wikipedia.org/wiki/Kernel_(image_processing)

    Author: Michael de Winter, Jeroen Esseveld
"""
class nso_tif_kernel_generator:

    """
        This class set up a .tif image in order to easily extracts kernel from it.
        With various parameters to control the size of the kernel.

        Fading, which means giving less weight to pixels other than the centre kernel, is also implemented here. 

    """

    def __init__(self,path_to_tif_file: str, x_size: int , y_size: int):
        """ 
        
        Init of the nso tif kernel.

        @param path_to_file: A path to a .tif file.
        @param x_size: the x size of the kernel. For example if x and y are 32 you get a 32x32 kernel.
        @param y_size: the y size of the kernel. For example if x and y are 32 you get a 32x32 kernel.
        """

        dataset = rasterio.open(path_to_tif_file)
        meta = dataset.meta.copy()
        data = dataset.read()
        width, height = meta["width"], meta["height"]

        self.data = data

        # TODO: Because of multiprocessing we can't stored rasterio datasets.
        self.path_to_tif_file = path_to_tif_file
        #self.dataset = dataset
        self.width = width
        self.height = height

        self.x_size = x_size 
        self.x_size_begin = round(x_size/2)
        self.x_size_end = round(x_size/2)

        self.y_size = y_size
        self.y_size_begin = round(y_size/2)
        self.y_size_end = round(y_size/2)


    def set_fade_kernel(self, fade_power = 0.045, bands = 4):
        """
        Creates a fading kernel based on the shape of the other kernels.

        @param fade_power: the power of the fade kernel.
        @param bands: the number bands that has to be faded.
        """

        self.fade_kernel = np.array([[(1-(fade_power*max(abs(idx-15),abs(idy-15)))) for idx in range(0,self.x_size)] for idy in range(0,self.y_size)])      
        self.fade_kernel = np.array([self.fade_kernel for id_x in range(0,bands)])

    def fade_tile_kernel(self, kernel):
        """

        Multiply a kernel with the fade kernel, thus fading it.

        @param kernel: A kernel you which to  fade.
        @return: A kernel that is faded now.
        """
        return kernel*self.fade_kernel

    def unfade_tile_kernel(self, kernel):
        """
        Unfade a kernel, for example to plot it again.

        @param kernel: A faded kernel that can be unfaded.
        @return: A unfaded kernel.
        """
        return kernel/self.fade_kernel


    def get_kernel_for_x_y(self,index_x,index_y):
        """

        Get a kernel with x,y as it's centre pixel.
        Be aware that the x,y coordinates have to be in the same coordinate system as the coordinate system in the .tif file.

        @param index_x: the x coordinate.
        @param index_y: the y coordinate.
        @return a kernel with chosen size in the init parameters
        """
        
        if sum([band[index_x][index_y] for band in self.data]) == 0:
            raise ValueError("Center pixel is empty")
        else:
            spot_kernel = [[k[index_y-self.x_size_end:index_y+self.x_size_begin] for k in band[index_x-self.y_size_end:index_x+self.y_size_begin] ] for band in self.data]
            spot_kernel = np.array(spot_kernel)
            spot_kernel = spot_kernel.astype(int)
            return spot_kernel



    def get_x_y(self, x_cor, y_cor):
        """
        
        Get the x and y, which means the x row and y column position in the matrix, based on the x, y in the geography coordinate system.
        Needed to get a kernel for a specific x and y in the coordinate system.

        Due to multi processing we have to read in the rasterio data set each time. 

        @param x_cor: x coordinate in the geography coordinate system.
        @param y_cor: y coordinate inthe geography coordinate system.
        @return x,y row and column position the matrix.
        """
        # TODO: Because of multi processing we have to read in the .tif every time.
        index_x, index_y = rasterio.open(self.path_to_tif_file).index(x_cor, y_cor)

        
        return pd.Series({'rd_x':int(index_x), 'rd_y':int(index_y)}) 


    def get_x_cor_y_cor(self, index_x , index_y):
        """
        Returns the geometry coordinates for index_x row and index_y column.
        
        @param index_x: the row.
        @param index_y: the column.
        """
        x_cor, y_cor = self.dataset.xy(index_x, index_y)
        return x_cor, y_cor


 

    def predict_all_output(self, amodel, output_location, aggregate_output = True, steps = 10, begin_part = 0):
        """
            Predict all the pixels in the .tif file.

            @param amodel: A prediciton model.
        """

        total_height = self.get_height() - self.x_size

        height_steps = round(total_height/steps)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin+height_steps

        total_height = self.get_height()-self.x_size
        total_width = self.get_width()-self.y_size

        height_steps = total_height/steps

        for x_step in range(begin_part,steps):
            print("-------")
            print("Part: "+str(x_step+1)+" of "+str(steps))
            print(begin_height)
            print(end_height)
            
            seg_df = np.zeros((((end_height-begin_height)*(self.get_width()-self.y_size)),3))
            seg_df_idx = 0
            for x in tqdm(range(begin_height, end_height)):
                for y in range(self.y_size_begin, self.get_width()-self.y_size_end):
                    
                    try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        actual_cor = self.get_x_cor_y_cor(x,y)  
                        kernel = self.get_kernel_for_x_y(x,y)
                        seg_df[seg_df_idx] = [actual_cor[0], actual_cor[1], amodel.predict(kernel)]
                        seg_df_idx = seg_df_idx+1

                    except ValueError as e:
                        xio = 0
                    except Exception as e:
                        print(e)

                    
            seg_df = pd.DataFrame(seg_df, columns = ['rd_x','rd_y','class'] )
            seg_df = seg_df[(seg_df['rd_x'] != 0) & (seg_df['rd_y'] != 0)]
            seg_df['class'] = seg_df.apply(lambda x: amodel.get_class_label(x['class']), axis=1)

            if aggregate_output == True:
                seg_df["x_group"] = np.round(seg_df["rd_x"]/2)*2
                seg_df["y_group"] = np.round(seg_df["rd_y"]/2)*2
                seg_df = seg_df.groupby(["x_group", "y_group"]).agg(label  = ('class', \
                                                        lambda x: x.value_counts().index[0])
                                                    )
            seg_df["x"] = list(map(lambda x: x[0], seg_df.index))
            seg_df["y"] = list(map(lambda x: x[1], seg_df.index))
            seg_df= seg_df[["x","y","label"]].values
            
            local_path_geojson = "./current.geojson"
            nso_ds_output.produce_geojson(seg_df,local_path_geojson)
            nso_ds_output.dissolve_label_geojson(local_path_geojson, output_location.replace(".","_part_"+str(x_step)+"."))
            print(output_location.replace(".","_part_"+str(x_step)+"."))
            os.remove(local_path_geojson)

            begin_height = int(round(end_height+1))
            end_height = int(round(begin_height+height_steps))
        
            if end_height > self.get_height() - (self.x_size/2):
                end_height = round(self.get_height() - (self.x_size/2))
        
        all_part = 0
        first_check = 0

        for file in glob.glob(output_location.replace(".","_part_*.")):
                        print(file)
                        if first_check == 0:
                            all_part = gpd.read_file(file)
                            first_check = 1
                        else:
                            print("Append")
                            all_part = all_part.append(gpd.read_file(file))
                        os.remove(file)

        all_part.dissolve(by='label').to_file(output_location)

        for file in glob.glob(output_location.replace(".","_part_*.").split(".")[0]):
            os.remove(file)


    def func_multi_processing(self, input_x_y):

         try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        #actual_cor = self.get_x_cor_y_cor(x,y)  
                        kernel = self.get_kernel_for_x_y(input_x_y[0],input_x_y[1])
                        return [input_x_y[0], input_x_y[1], self.model.predict(kernel)]

         except ValueError as e:
                        return [0,0,0]
         except Exception as e:
                        print(e)
                        return [0,0,0]


    def predict_all_output_multiprocessing(self, amodel, output_location, aggregate_output = True, steps = 10, begin_part = 0):
        """
            Predict all the pixels in the .tif file.

            @param amodel: A prediciton model.
        """

        total_height = self.get_height() - self.x_size

        height_steps = round(total_height/steps)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin+height_steps

        total_height = self.get_height()-self.x_size
        total_width = self.get_width()-self.y_size

        height_steps = total_height/steps

        
        self.set_model(amodel)

        for x_step in tqdm(range(begin_part,steps)):
            print("-------")
            print("Part: "+str(x_step+1)+" of "+str(steps))
            permutations = list(itertools.product([x for x in range(begin_height, end_height)], [ y for y in range(self.y_size_begin, self.get_width()-self.y_size_end)]))
            print(" Total permutations this step: "+str(len(permutations)))
            
            start = timer() 
            p = Pool()
            seg_df = p.map(self.func_multi_processing,permutations)
            p.terminate()
            print("Pool finised in: "+str(timer()-start)+" second(s)")
         
            start = timer() 
            seg_df = pd.DataFrame(seg_df, columns = ['x_cor','y_cor','class'] )
            seg_df = seg_df[(seg_df['x_cor'] != 0) & (seg_df['y_cor'] != 0)]
            seg_df['class'] = seg_df.apply(lambda x: amodel.get_class_label(x['class']), axis=1)
            seg_df = pd.concat([seg_df, seg_df.apply(lambda x: self.get_x_y(x['x_cor'], x['y_cor'] ),axis=1)], axis='columns')
            seg_df = seg_df.drop(['y_cor','x_cor'], axis=1)
            print("Pandas filter finised in: "+str(timer()-start)+" second(s)")

            start = timer() 
            if aggregate_output == True:
                seg_df["x_group"] = np.round(seg_df["rd_x"]/2)*2
                seg_df["y_group"] = np.round(seg_df["rd_y"]/2)*2
                seg_df = seg_df.groupby(["x_group", "y_group"]).agg(label  = ('class', \
                                                        lambda x: x.value_counts().index[0])
                                                    )
            seg_df["x"] = list(map(lambda x: x[0], seg_df.index))
            seg_df["y"] = list(map(lambda x: x[1], seg_df.index))
            seg_df= seg_df[["x","y","label"]].values
            
            local_path_geojson = "./current.geojson"
            nso_ds_output.produce_geojson(seg_df,local_path_geojson)
            nso_ds_output.dissolve_label_geojson(local_path_geojson, output_location.replace(".","_part_"+str(x_step)+"."))
            print(output_location.replace(".","_part_"+str(x_step)+"."))
            os.remove(local_path_geojson)
            print("Writing finised in: "+str(timer()-start)+" second(s)")

            begin_height = int(round(end_height+1))
            end_height = int(round(begin_height+height_steps))
        
            if end_height > self.get_height() - (self.x_size/2):
                end_height = round(self.get_height() - (self.x_size/2))
        
        all_part = 0
        first_check = 0

        for file in glob.glob(output_location.replace(".","_part_*.")):
                        print(file)
                        if first_check == 0:
                            all_part = gpd.read_file(file)
                            first_check = 1
                        else:
                            print("Append")
                            all_part = all_part.append(gpd.read_file(file))
                        os.remove(file)

        all_part.dissolve(by='label').to_file(output_location)

        for file in glob.glob(output_location.replace(".","_part_*.").split(".")[0]):
            os.remove(file)


    def set_model(self, amodel):
        self.model = amodel        

    def get_height(self):
        """
        Get the height of the .tif file.

        @return the height of the .tif file.
        """
        return self.height

    def get_width(self):
        """
        Get the width of the .tif file.

        @return the width of the .tif file.
        """
        return self.width

    def get_data(self):
        """
        
        Return the numpy array with all the spectral data in it.

        @return the numpy data with the spectral data  in it.
        """
        return self.data


def plot_kernel(kernel,y=0 ):
        """
        Plot a kernel or .tif image.
        
        Multiple inputs are correct either a numpy array or x,y coordinates.

        @param kernel: A kernel that you want to plot or x coordinate.
        @param y: the y coordinate you want to plot.
        """

        if isinstance(kernel, int):
            rasterio.plot.show(np.clip(self.get_kernel_for_x_y(kernel,y)[2::-1],0,2200)/2200 )
        else:
            rasterio.plot.show(np.clip(kernel[2::-1],0,2200)/2200 )



