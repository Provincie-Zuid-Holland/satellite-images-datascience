from multiprocessing import Pool
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
from nso_ds_classes.nso_ds_models import euclidean_distance_kernels
from os.path import exists
from random import sample




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
            """
                
            
            """

            annotations_stats = pd.read_csv("./annotations/median_stats_annotations.csv")
            annotations_stats = annotations_stats[["MEDIAN_band1_normalized","MEDIAN_band2_normalized","MEDIAN_band3_normalized","MEDIAN_band4_normalized","MEDIAN_height_normalized","MEDIAN_ndvi_normalized", "Label"]]

            cluster_centers = self.retrieve_stepped_cluster_centers(parts=2,begin_part=1, output_name = a_output_name)
            cluster_centers_df = pd.DataFrame(cluster_centers[0], columns =["band3","band5","band6"])



            cluster_centers_df["labels"] =cluster_centers_df.apply\
                    (lambda row_stuff:annotations_stats['Label'][annotations_stats[["MEDIAN_band3_normalized","MEDIAN_height_normalized","MEDIAN_ndvi_normalized"]].apply(lambda x:euclidean_distance_kernels(row_stuff,x.values),axis=1).idxmin()], axis=1
            )

            cluster_centers_df.to_csv(a_output_name)

            return cluster_centers_df

        def get_random_samples(a, b, n):
            """
            Generate n samples from product a,b

            @param a: a a array with numbers.
            @param b: a b array with numbers.
            @return the sampled permutations of the product of a and b.
            """

            n_prod = len(a) * len(b)
            indices = sample(range(n_prod), n)
            return [(a[idx % len(a)], b[idx // len(a)]) for idx in indices]


        def retrieve_stepped_cluster_centers(self,output_name = ""):

                #TODO: Fix making a cluster center.
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
