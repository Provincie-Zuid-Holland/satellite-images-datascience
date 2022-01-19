import numpy as np 
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import pandas as pd

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
        self.dataset = dataset
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
            raise "Centre pixel is empty"
        else:
            spot_kernel = [[k[index_y-self.x_size_end:index_y+self.x_size_begin] for k in band[index_x-self.y_size_end:index_x+self.y_size_begin] ] for band in self.data]
            spot_kernel = np.array(spot_kernel)
            spot_kernel = spot_kernel.astype(int)
            return spot_kernel



    def get_x_y(self, x_cor, y_cor):
        """
        
        Get the x and y, which means the x row and y column position in the matrix, based on the x, y in the geography coordinate system.
        Needed to get a kernel for a specific x and y in the coordinate system.

        @param x_cor: x coordinate in the geography coordinate system.
        @param y_cor: y coordinate inthe geography coordinate system.
        @return x,y row and column position the matrix.
        """
        index_x, index_y = self.dataset.index(x_cor, y_cor)
        return index_x,index_y

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



