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
import multiprocessing
from multiprocessing import Pool
import itertools
from timeit import default_timer as timer
from shapely.geometry import Polygon
from sklearn import preprocessing




"""
    This code is used to extract image processing kernels from nso satellite images .tif images

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
        @param x_size: the x size of the kernel. For example if x and y are 32 you get a 32 by y kernel.
        @param y_size: the y size of the kernel. For example if x and y are 32 you get a x by 32 kernel.
        """

        dataset = rasterio.open(path_to_tif_file)
        meta = dataset.meta.copy()
        data = dataset.read()
        width, height = meta["width"], meta["height"]

        self.data = data

        self.bands = data.shape[0]
        
        self.path_to_tif_file = path_to_tif_file

        # TODO: Because of multiprocessing we can't stored rasterio datasets.
        #self.dataset = dataset
        self.width = width
        self.height = height

        self.x_size = x_size  
        self.x_size_begin = round(x_size/2) 
        self.x_size_end = round(x_size/2) 

        self.y_size = y_size
        self.y_size_begin = round(y_size/2)
        self.y_size_end = round(y_size/2)

        self.sat_name = path_to_tif_file.split("/")[-1]


    def set_fade_kernel(self, fade_power = 0.045, bands = 0):
        """
        Creates a fading kernel based on the shape of the other kernels and different parameters.

        A fading kernel uses weights to give other pixel than the center pixel less weight in the prediction.

        @param fade_power: the power of the fade kernel.
        @param bands: the number bands that has to be faded.
        """
        if bands == 0: 
            bands = self.data.shape[0]

        self.fade_kernel = np.array([[(1-(fade_power*max(abs(idx-15),abs(idy-15)))) for idx in range(0,self.x_size)] for idy in range(0,self.y_size)])      
        self.fade_kernel = np.array([self.fade_kernel for id_x in range(0,bands)])

    def fadify_kernel(self, kernel):
        """

        Multiply a kernel with the fade kernel, thus fading it.

        A fading kernel uses weights to give other pixel than the center pixel less weight in the prediction.

        @param kernel: A kernel you which to  fade.
        @return: A kernel that is faded now.
        """
        return kernel*self.fade_kernel



    def normalize_tile_kernel(self, kernel):
        """
            Normalize image kernels with sklearn's normalize.

            @param kernel: a kernel to normalize
        
        """
        
        copy_kernel = np.zeros(shape=kernel.shape)
        for x in range(0,kernel.shape[0]):
            copy_kernel[x] = preprocessing.normalize(kernel[x]) 

        return copy_kernel

    def unfadify_tile_kernel(self, kernel):
        """
        Unfade a kernel, for example to plot it again.

        A fading kernel uses weights to give other pixel than the center pixel less weight in the prediction.

        @param kernel: A faded kernel that can be unfaded.
        @return: A unfaded kernel.
        """
        return kernel/self.fade_kernel

    def get_pixel_value(self,index_x,index_y):


        if sum([band[index_x][index_y] for band in self.data]) == 0:
            raise ValueError("Center pixel is empty")
        else:
            return [band[index_x][index_y] for band in self.data]
            
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
            spot_kernel = [[k[index_y-self.x_size_end:index_y+self.x_size_begin] for k in band[index_x-self.y_size_end:index_x+self.y_size_begin]] for band in self.data]
            spot_kernel = np.array(spot_kernel)
            spot_kernel = spot_kernel.astype(int)
            return spot_kernel



    def get_x_y(self, x_cor, y_cor, dataset = False):
        """
        
        Get the x and y, which means the x row and y column position in the matrix, based on the x, y in the geography coordinate system.
        Needed to get a kernel for a specific x and y in the coordinate system.

        Due to multi processing we have to read in the rasterio data set each time. 

        @param x_cor: x coordinate in the geography coordinate system.
        @param y_cor: y coordinate inthe geography coordinate system.
        @return x,y row and column position the matrix.
        """
        # TODO: Because of multi processing we have to read in the .tif every time.
        if isinstance(dataset, bool):
            index_x, index_y = rasterio.open(self.path_to_tif_file).index(x_cor, y_cor)

        else:    
            index_x, index_y = dataset.index(x_cor, y_cor)
        
        return pd.Series({'rd_x':int(index_x), 'rd_y':int(index_y)}) 


    def get_x_cor_y_cor(self, index_x , index_y, dataset = False):
        """
        Returns the geometry coordinates for index_x row and index_y column.
        
        @param index_x: the row.
        @param index_y: the column.
        """
        if isinstance(dataset, bool):
            index_x, index_y = rasterio.open(self.path_to_tif_file).index(index_x, index_y)

        else:    
            index_x, index_y = dataset.xy(index_x, index_y)
        
        return pd.Series({'rd_x':int(index_x), 'rd_y':int(index_y)}) 


 

    def predict_all_output(self, amodel, output_location, aggregate_output = True, parts = 10, begin_part = 0):
        """
            Predict all the pixels in the .tif file.

            @param amodel: A prediciton model.
        """

        total_height = self.get_height() - self.x_size

        height_parts = round(total_height/parts)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin+height_parts

        total_height = self.get_height()-self.x_size
        total_width = self.get_width()-self.y_size

        height_parts = total_height/parts

        dataset = rasterio.open(self.path_to_tif_file)

        for x_step in range(begin_part,parts):
            print("-------")
            print("Part: "+str(x_step+1)+" of "+str(parts))
            print(begin_height)
            print(end_height)
            
            seg_df = np.zeros((((end_height-begin_height)*(self.get_width()-self.y_size)),3))
            seg_df_idx = 0
            for x in tqdm(range(begin_height, end_height)):
                for y in range(self.y_size_begin, self.get_width()-self.y_size_end):
                    
                    try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        actual_cor = self.get_x_cor_y_cor(x,y,dataset)  
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
            end_height = int(round(begin_height+height_parts))
        
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


    def func_multi_processing_get_kernels(self, input_x_y):
         """
            This function is used to do multiprocessing predicting.

            This needs to be done in a seperate function in order to make multiprocessing work.

            @param input_x_y: a array with the row and column for the to be predicted pixel.
            @return row and column and the predicted label in numbers.
         """
         try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        #actual_cor = self.get_x_cor_y_cor(x,y)  
                        # TODO: Maybe select bands in get_kernel_for_x_y
                        kernel = self.get_kernel_for_x_y(input_x_y[0],input_x_y[1]) if self.pixel_values == False else self.get_pixel_value(input_x_y[0],input_x_y[1])
                        
                        # TODO: Fix the extra checking.
                        #try:
                        #    kernel = np.array([ kernel[x-1] for x in self.bands])
                        #except Exception as e:
                        #    print(e)
                        #    print("No bands selected")
                        #kernel = self.normalize_tile_kernel(kernel) if self.normalize == True else kernel
                        #kernel = self.fadify_kernel(kernel) if self.fade == True else kernel        
                        
                        
                        return  kernel

         except ValueError as e:                  
                        if str(e) != "Center pixel is empty":                          
                            print(e)
                        return [0,0,0]
         except Exception as e:
                        print(e)
                        return [0,0,0]

    def func_multi_processing_predict(self, input_x_y):

        try:

            # TODO: Make the bands selected able
            label = self.model.predict([input_x_y[2],input_x_y[4], input_x_y[5]])
            return [input_x_y[6][0], input_x_y[6][1], label]

        except ValueError as e:                  
                        if str(e) != "Center pixel is empty":                          
                            print(e)
                        return [0,0,0]
        except Exception as e:
                        print(e)
                        return [0,0,0]
        
 
    def predict_all_output_multiprocessing(self, amodel, output_location, aggregate_output = True, parts = 10, begin_part = 0, bands = [1,2,3,4,5,6], fade = False, normalize = False, pixel_values = False ):
        """
            Predict all the pixels in the .tif file with a kernel per pixel.

            Uses multiprocessing to speed up the results.

            @param amodel: A prediciton model with has to have a predict function and uses kernels as input.
            @param output_location: Location where to writes the results too.
            @param aggregate_output: 50 cm is the default resolution but we can aggregate to 2m.
            @param parts: break the .tif file in multiple parts, this is needed because some .tif files can contain 3 billion pixels which won't fit in one pass in memory thus we divide a .tif file in multiple parts.
            @param begin_part: skip certain parts in the parts.
            @param bands: Which bands of the .tif file to use from the .tif file by default this will be all the bands.
            @param fade: Whether to use fading kernels or not.
            @param normalize: Whether to use normalize all the kernels or not.
        """
        #TODO: Export all variables to other sections.

        # Set some variables for breaking the .tif in different part parts in order to save memory.
        total_height = self.get_height() - self.x_size

        height_parts = round(total_height/parts)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin+height_parts

        total_height = self.get_height()-self.x_size
        total_width = self.get_width()-self.y_size

        height_parts = total_height/parts

        # Set some variables for multiprocessing.
        self.set_model(amodel)
        dataset = rasterio.open(self.path_to_tif_file)

        try:
            self.fade = amodel.get_fade()
            self.normalize = amodel.get_normalize()
        except:
            self.fade = fade
            self.normalize = normalize
        
        self.pixel_values = pixel_values 
        self.bands = bands

        # Loop through the parts.
        for x_step in tqdm(range(begin_part,parts)):
            print("-------")
            print("Part: "+str(x_step+1)+" of "+str(parts))
            # Calculate the number of permutations for this step.
            permutations = list(itertools.product([x for x in range(begin_height, end_height)], [ y for y in range(self.y_size_begin, self.get_width()-self.y_size_end)]))
            print("Total permutations this step: "+str(len(permutations)))
            
            # Init the multiprocessing pool.
            # TODO: Maybe use swifter for this?
            start = timer() 
            p = Pool()
            seg_df = p.map(self.func_multi_processing_get_kernels,permutations)
            print("Pool kernel fetching finised in: "+str(timer()-start)+" second(s)")
          
            

           
            seg_df = pd.DataFrame(seg_df, columns= ["band1","band2","band3","band4","band5","band6"])

            if normalize is not False:
                print("Normalizing data")
                seg_df = normalize.transform(seg_df)
            

            seg_df["permutation"] = permutations
            seg_df = seg_df.dropna()
            
            seg_df = seg_df.values
            del permutations
         
            start = timer() 

            seg_df = p.map(self.func_multi_processing_predict,seg_df)

            print("Predicting finised in: "+str(timer()-start)+" second(s)")

            seg_df = pd.DataFrame(seg_df, columns = ['x_cor','y_cor','label'])
            seg_df = seg_df[(seg_df['x_cor'] != 0) & (seg_df['y_cor'] != 0)]
            print("Number of used pixels for this step: "+str(len(seg_df)))

            # Get the coordinates for the pixel locations.           
            seg_df['rd_x'],seg_df['rd_y'] = rasterio.transform.xy(dataset.transform,seg_df['x_cor'], seg_df['y_cor'])
            
            print("Got coordinates for pixels: "+str(timer()-start)+" second(s)")

            seg_df = seg_df.drop(['y_cor','x_cor'], axis=1)
            

            start = timer() 
            if aggregate_output == True:
                seg_df["x_group"] = np.round(seg_df["rd_x"]/2)*2
                seg_df["y_group"] = np.round(seg_df["rd_y"]/2)*2
                seg_df = seg_df.groupby(["x_group", "y_group"]).agg(label  = ('label', \
                                                        lambda x: x.value_counts().index[0]))
                print("Group by finised in: "+str(timer()-start)+" second(s)")
                
                start = timer() 
                seg_df["rd_x"] = list(map(lambda x: x[0], seg_df.index))
                seg_df["rd_y"] = list(map(lambda x: x[1], seg_df.index))
                print("Labels created in: "+str(timer()-start)+" second(s)")
                
                seg_df= seg_df[["rd_x","rd_y","label"]]

            start = timer()  
            
            # Make squares from the the pixels in order to make contected polygons from them.
            seg_df['geometry'] = p.map(func_cor_square, seg_df[["rd_x","rd_y"] ].to_numpy().tolist())
            
            p.terminate()
            seg_df= seg_df[["geometry","label"]]

            # Store the results in a geopandas dataframe.
            seg_df = gpd.GeoDataFrame(seg_df, geometry=seg_df.geometry)
            seg_df = seg_df.set_crs(epsg = 28992)
            print("Geometry made in: "+str(timer()-start)+" second(s)")
            nso_ds_output.dissolve_gpd_output(seg_df, output_location.replace(".","_part_"+str(x_step)+"."))
            print(output_location.replace(".","_part_"+str(x_step)+"."))

            print("Writing finised in: "+str(timer()-start)+" second(s)")
            print(seg_df.columns)
            del seg_df
            begin_height = int(round(end_height+1))
            end_height = int(round(begin_height+height_parts))
        
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
                            
                       

        try:
            if str(type(amodel)) != "<class 'nso_ds_classes.nso_ds_models.deep_learning_model'>" or str(type(amodel)) == "<class 'nso_ds_classes.nso_ds_models.waterleiding_ahn_ndvi_model'>":
                all_part['label'] = all_part.apply(lambda x: amodel.get_class_label(x['label']), axis=1)
        except Exception as e:
            print(e)
            
        all_part.dissolve(by='label').to_file(output_location)
        
        for file in glob.glob(output_location.replace(".","_part_*.").split(".")[0]):
            os.remove(file)

    def get_kernel_multi_processing(self, input_x_y):
         """
            This function is used to do multiprocessing predicting.

            This will get all the kernels first to be predicted later with a keras prediction function.
            Keras performs better when you give it multiple inputs instead of one.

            @param input_x_y: a array with the row and column for the to be predicted pixel.
            @return row and column and the kernel.
         """
         try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        #actual_cor = self.get_x_cor_y_cor(x,y)  
                        kernel = self.get_kernel_for_x_y(input_x_y[0],input_x_y[1])
                        # TODO: Set normalisation if used.
                        #kernel = self.normalize_tile_kernel(kernel) if self.normalize == True else kernel
                                         
                        return [input_x_y[0], input_x_y[1], kernel]

         except ValueError as e:                  
                        if str(e) != "Center pixel is empty":                          
                            print(e)
                        #return [0,0,0]
         except Exception as e:
                        print(e)
                        #return [0,0,0]

    def predict_keras_multi_processing(self, input_x_y_kernel):
        """
            This function is used to do multiprocessing predicting.

            Prediction function for keras models

            @param input_x_y: a array of kernels for keras predict to use.
            @return row and column and the predicted label in numbers.
         """
        try:
                
                # Fetches the real coordinates for the row and column needed for writing to a geoformat.               
                kernels = [arow[2] for arow  in input_x_y_kernel]
                
                # TODO: Fix bands and labels
                predicts = self.model.predict(kernels)
                print(predicts)

                row_id = 0
                returns = []
                for input_row in  input_x_y_kernel:
                    returns.append([input_row[0], input_row[1], predicts[row_id]])

                return returns

        except ValueError as e:
                        print("Error in multiprocessing prediction:")
                        print(e)
                        #return [0,0,0]
        except Exception as e:
                        print("Error in multiprocessing prediction:")
                        print(e)
                        #return [0,0,0]

    def predict_all_output_multiprocessing_keras(self, amodel, output_location, aggregate_output = True, parts = 10, begin_part = 0, keras_break_size = 10000,  multiprocessing = False):
        """
            Predict all the pixels in the .tif file with kernels per pixel.

            Uses multiprocessing to speed up the results.

            @param amodel: A prediciton model with has to have a predict function.
            @param output_location: Locatie where to writes the results too.
            @param aggregate_output: 50 cm is the default resolution but we can aggregate to 2m
            @param parts: break the .tif file in multiple parts this is needed because some .tif files can contain 3 billion pixels which won't fit in one pass in memory.
            @param begin_part: skip certain parts in the parts
        """
        
        # Set some variables for breaking the .tif in different part parts in order to save memory.
        total_height = self.get_height() - self.x_size

        height_parts = round(total_height/parts)
        begin_height = self.x_size_begin
        end_height = self.x_size_begin+height_parts

        total_height = self.get_height()-self.x_size
        total_width = self.get_width()-self.y_size

        height_parts = total_height/parts

        # Set some variables for multiprocessing.
        self.set_model(amodel)
        dataset = rasterio.open(self.path_to_tif_file)
     
        #TODO: Set normalisation if used.
        #self.normalize = amodel.get_normalize()


        self.keras_break_size = keras_break_size 
        
        # Loop through the parts.
        for x_step in tqdm(range(begin_part,parts)):
            print("-------")
            print("Part: "+str(x_step+1)+" of "+str(parts))
            # Calculate the number of permutations for this step.
            permutations = list(itertools.product([x for x in range(begin_height, end_height)], [ y for y in range(self.y_size_begin, self.get_width()-self.y_size_end)]))


            permutations = np.array(permutations)

            print("Total permutations this step: "+str(len(permutations)))
            
            # Init the multiprocessing pool.
            # TODO: Maybe use swifter for this?
            start = timer() 
            print("Getting kernels")
          

            if multiprocessing == True:
               
                p = Pool()
                permutations = np.array(p.map(self.get_kernel_multi_processing, permutations))
                permutations = permutations[permutations != None]
                p.terminate()

                print("kernels at first step:")
                original_shape = permutations.shape[0]
                print(permutations.shape)

                permutations = np.array_split(permutations,self.keras_break_size)
                print("after split")
                print(len(permutations))
                #print("break size: "+ str(keras_break_size ))
                p = Pool()
                permutations = p.map(self.predict_keras_multi_processing,permutations)
                p.terminate()
           
            
         
            else:

              
                permutations = np.array([self.get_kernel_multi_processing(permutation) for permutation in permutations],dtype='object')
                print(permutations)
                #permutations = permutations[permutations != None]
                print("kernels at first step:")
                original_shape = permutations.shape[0]
                print(permutations.shape)
                array_split_size = round(permutations.shape[0]/self.keras_break_size)
                permutations = np.array_split(permutations, array_split_size)
            
                print("After split")
                print(len(permutations))
                print("With size:")
                print(len(permutations[0]))
                print("Predicting")
                permutations = [self.predict_keras_multi_processing(kernels) for kernels in permutations]
                print("After predict")
                print(len(permutations))
                print(permutations)

            try:
                permutations = np.concatenate(permutations)
                permutations = permutations.reshape(original_shape,3)

                
                print("Pool finised in: "+str(timer()-start)+" second(s)")
            
            
                start = timer() 
                seg_df = pd.DataFrame(permutations, columns = ['x_cor','y_cor','label'])
                del permutations
                seg_df = seg_df[(seg_df['x_cor'] != 0) & (seg_df['y_cor'] != 0)]
                print(seg_df)
                print("Number of used pixels for this step: "+str(len(seg_df)))
                
                if len(seg_df) > 0:
                    

                    # Get the coordinates for the pixel locations.           
                    seg_df['rd_x'],seg_df['rd_y'] = rasterio.transform.xy(dataset.transform,seg_df['x_cor'], seg_df['y_cor'])
                    
                    print("Got coordinates for pixels: "+str(timer()-start)+" second(s)")

                    seg_df = seg_df.drop(['y_cor','x_cor'], axis=1)
                    

                    start = timer() 
                    if aggregate_output == True:
                        seg_df["x_group"] = np.round(seg_df["rd_x"]/2)*2
                        seg_df["y_group"] = np.round(seg_df["rd_y"]/2)*2
                        seg_df = seg_df.groupby(["x_group", "y_group"]).agg(label  = ('label', \
                                                                lambda x: x.value_counts().index[0]))
                        print("Group by finised in: "+str(timer()-start)+" second(s)")
                        
                        start = timer() 
                        seg_df["rd_x"] = list(map(lambda x: x[0], seg_df.index))
                        seg_df["rd_y"] = list(map(lambda x: x[1], seg_df.index))
                        print("Labels created in: "+str(timer()-start)+" second(s)")
                        
                        seg_df= seg_df[["rd_x","rd_y","label"]]

                    start = timer()  
                    
                    # Make squares from the the pixels in order to make contected polygons from them.
                    p = Pool()
                    seg_df['geometry'] = p.map(func_cor_square, seg_df[["rd_x","rd_y"] ].to_numpy().tolist())
                    
                    p.terminate()
                    seg_df= seg_df[["geometry","label"]]

                    # Store the results in a geopandas dataframe.
                    seg_df = gpd.GeoDataFrame(seg_df, geometry=seg_df.geometry)
                    seg_df = seg_df.set_crs(epsg = 28992)
                    print("Geometry made in: "+str(timer()-start)+" second(s)")
                    try:
                        nso_ds_output.dissolve_gpd_output(seg_df, output_location.replace(".","_part_"+str(x_step)+"."))
                        print(output_location.replace(".","_part_"+str(x_step)+"."))
                    except:
                        print("Warning nothing has been written")

                    print("Writing finised in: "+str(timer()-start)+" second(s)")
                    print(seg_df.columns)
                    del seg_df
                    begin_height = int(round(end_height+1))
                    end_height = int(round(begin_height+height_parts))
                
                    if end_height > self.get_height() - (self.x_size/2):
                        end_height = round(self.get_height() - (self.x_size/2))
                else:
                    print("WARNING! Empty DataFrame!")
            except Exception as e:
                print(e)
        
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
                            
                       

   
       
        all_part.dissolve(by='label').to_file(output_location)
        
        for file in glob.glob(output_location.replace(".","_part_*.").split(".")[0]):
            os.remove(file)

    def set_model(self, amodel):
        """ 
        Set a model coupled to this .tif generator.
        Mostly used for multiprocessing purposes

        @param amodel: The specific model to set.
        
        """
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

    def get_sat_name(self):
        """
        
        Return the satellite name based on the file extension.

        @return string with the satellite name.
        """

        return self.sat_name

def normalizedata(data):
        """
        Normalize between 0 en 1.


        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_kernel(kernel,y=0 ):
        """
        Plot a kernel or .tif image.
        
        Multiple inputs are correct either a numpy array or x,y coordinates.

        @param kernel: A kernel that you want to plot or x coordinate.
        @param y: the y coordinate you want to plot.
        """

        if isinstance(kernel, int):
            rasterio.plot.show(np.clip(self.get_kernel_for_x_y(kernel,y)[2::-1],0,2200)/2200)
        else:
            rasterio.plot.show(np.clip(kernel[2::-1],0,2200)/2200)

def func_cor_square(input_x_y):

        rect = [round(input_x_y[0]/2)*2, round(input_x_y[1]/2)*2, 0, 0]
        rect[2], rect[3] = rect[0] + 2, rect[1] + 2
        coords = Polygon([(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]), (rect[0], rect[1])])
        return coords


