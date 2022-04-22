import multiprocessing
import rasterio
import tqdm
import itertools
from multiprocessing import Pool
from sklearn.cluster import KMeans
import numpy as np 
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nso_ds_classes.nso_ds_models import euclidean_distance_kernels
import joblib
from os.path import exists


class normalize_scaler_class_BNDVIH:
    
    def __init__(self, scaler_file_band3 = "",scaler_file_band5 = "", scaler_file_band6 = "") :
        
        self.scaler_band3 = joblib.load(scaler_file_band3)
        self.scaler_band5 = joblib.load(scaler_file_band5)
        self.scaler_band6 = joblib.load(scaler_file_band6)

    def transform(pixel_df):
        


class nso_cluster_break:


        def __init__(self, a_kernel_generator):
            self.kernel_generator =  a_kernel_generator

        def get_pixel_multiprocessing(self, input_x_y):
            try:
                        # Fetches the real coordinates for the row and column needed for writing to a geoformat.
                        #actual_cor = self.get_x_cor_y_cor(x,y)  
                       
                        # TODO: Set normalisation if used.
                        #kernel = self.normalize_tile_kernel(kernel) if self.normalize == True else kernel
                                         
                        return self.kernel_generator.get_pixel_value( input_x_y[0],  input_x_y[1])

            except Exception as e:
                if str(e) != "Center pixel is empty":                          
                            print(e)

        def make_clusters_centers(self, a_output_name):

            annotations_stats = pd.read_csv("./annotations/median_stats_annotations.csv")
            annotations_stats = annotations_stats[["MEDIAN_band1_normalized","MEDIAN_band2_normalized","MEDIAN_band3_normalized","MEDIAN_band4_normalized","MEDIAN_height_normalized","MEDIAN_ndvi_normalized", "Label"]]

            cluster_centers = self.retrieve_stepped_cluster_centers(steps=2,begin_part=1, output_name = a_output_name)
            cluster_centers_df = pd.DataFrame(cluster_centers[0], columns =["band3","band5","band6"])



            cluster_centers_df["labels"] =cluster_centers_df.apply\
                    (lambda row_stuff:annotations_stats['Label'][annotations_stats[["MEDIAN_band3_normalized","MEDIAN_height_normalized","MEDIAN_ndvi_normalized"]].apply(lambda x:euclidean_distance_kernels(row_stuff,x.values),axis=1).idxmin()], axis=1
            )

            cluster_centers_df.to_csv(a_output_name)

            return cluster_centers_df




        def get_stepped_pixel_df(self,steps=10, begin_part=0, multiprocessing = False, output_name = ""):

            total_height = self.kernel_generator.get_height() - self.kernel_generator.x_size

            height_steps = round(total_height/steps)
            begin_height = self.kernel_generator.x_size_begin
            end_height = self.kernel_generator.x_size_begin+height_steps

            total_height = self.kernel_generator.get_height()-self.kernel_generator.x_size
            total_width = self.kernel_generator.get_width()-self.kernel_generator.y_size

            height_steps = total_height/steps

            clusters_centers = []
            # Loop through the steps.
            for x_step in tqdm.tqdm(range(begin_part,steps)):
                print("-------")
                print("Part: "+str(x_step+1)+" of "+str(steps))
                # Calculate the number of permutations for this step.
                permutations = list(itertools.product([x for x in range(begin_height, end_height)], [ y for y in range(self.kernel_generator.y_size_begin, self.kernel_generator.get_width()-self.kernel_generator.y_size_end)]))
                print("Total permutations this step: "+str(len(permutations)))


                print("Retrieving kernels:")
                if multiprocessing == True:
                    p = Pool()
                    pixel_df = p.map(self.get_pixel_multiprocessing ,permutations)
                    p.terminate()
                else:
                    pixel_df = [ self.get_pixel_multiprocessing(permutation) for permutation in permutations]

                print("Number of pixels:")
                print(len(pixel_df))
                
                pixel_df= [elem for elem in pixel_df if elem is not None]

                pixel_df = pd.DataFrame(pixel_df, columns= ["band1","band2","band3","band4","band5","band6"])
                pixel_df = self.make_normalized_clusters(pixel_df, output_name)

            return pixel_df

        def make_normalized_clusters(self, pixel_df,output_name ):



            band3_scaler = MinMaxScaler().fit(pixel_df['band3'].values.reshape(-1, 1))
            joblib.dump(band3_scaler,"./scalers/"+output_name+"_band3.save") 
            pixel_df['band3'] = band3_scaler.transform(pixel_df['band3'].values.reshape(-1, 1))
            

            band5_scaler = MinMaxScaler().fit(pixel_df['band5'].values.reshape(-1, 1))
            joblib.dump(band5_scaler, "./scalers/"+output_name+"_band5.save") 
            pixel_df['band5'] = band5_scaler.transform(pixel_df['band5'].values.reshape(-1, 1))


            ahn4_scaler = "./scalers/ahn4.save"
            if exists(ahn4_scaler):
                 
                    band6_scaler= joblib.load(ahn4_scaler) 
            else:
                    band6_scaler = MinMaxScaler().fit(pixel_df['band6'].values.reshape(-1, 1))               
                    joblib.dump(band6_scaler,"./scalers/ahn4.save") 

            pixel_df['band6'] = band6_scaler.transform(pixel_df['band6'].values.reshape(-1, 1))

            return pixel_df


        def retrieve_stepped_cluster_centers(self,output_name = ""):

            
                #pixel_df['band1'] =MinMaxScaler().fit_transform(pixel_df['band1'].values.reshape(-1, 1) )
                #pixel_df['band2'] =MinMaxScaler().fit_transform(pixel_df['band2'].values.reshape(-1, 1) )
               
                
                clusters_centers = []
                try:
                    pixel_df = pixel_df[['band3','band5', 'band6']].values

                    model = KMeans()
                    visualizer = KElbowVisualizer(model, k=(4,12))

                    visualizer.fit(pixel_df) 
                    #visualizer.show() 

                    model = KMeans(n_clusters=visualizer.elbow_value_)
                    

                    clusters_centers.append(model.fit(pixel_df).cluster_centers_)
                except Exception as e:
                    print(e)
            
                return clusters_centers
