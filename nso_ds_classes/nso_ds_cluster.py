import multiprocessing
import rasterio
import tqdm
import itertools
from multiprocessing import Pool
from sklearn.cluster import KMeans
import numpy as np 
from yellowbrick.cluster import KElbowVisualizer


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


        def retrieve_stepped_cluster_centers(self,steps=10, begin_part=0, multiprocessing = False):

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
                else:
                    pixel_df = [ self.get_pixel_multiprocessing(permutation) for permutation in permutations]

                print("Number of pixels:")
                print(len(pixel_df))
                
                pixel_df= [elem for elem in pixel_df if elem is not None]

                try:
                    pixel_df = np.array(pixel_df)

                    model = KMeans()
                    visualizer = KElbowVisualizer(model, k=(4,12))

                    visualizer.fit(pixel_df) 
                    visualizer.show() 

                    model = KMeans(n_clusters=visualizer.elbow_value_)
                    

                    clusters_centers.append(model.fit(pixel_df).cluster_centers_)
                except Exception as e:
                    print(e)
            
            return clusters_centers
