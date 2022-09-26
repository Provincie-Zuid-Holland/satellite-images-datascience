
import joblib
import pandas as pd
import itertools
import tqdm
from multiprocessing import Pool

class scaler_class_BNDVIH:
    """
    This class is used to scale blue, ndvi and height columns of a pandas dataframe from a .tif file.
    Which should be band 3, band 5 and band 6 respectively.

    Scalers should have been made indepently!
    
    """
    def __init__(self, scaler_file_band3 = "",scaler_file_band5 = "", scaler_file_band6 = "") :
        """
        Init of this class.

        @param scaler_file_band3: Path to a file which contains the scaler for band 3.
        @param scaler_file_band5: Path to a file which contains the scaler for band 5.
        @param scaler_file_band6: Path to a file which contains the scaler for band 6.
        
        """
        self.scaler_band3 = joblib.load(scaler_file_band3)
        self.scaler_band5 = joblib.load(scaler_file_band5)
        self.scaler_band6 = joblib.load(scaler_file_band6)

    def transform(self,pixel_df, col_names = ['band3','band5',"band6"]):
        """
        Transforms the blue, ndvi and height columns of a pandas dataframe.

        @param pixel_df: dataframe in which the blue, ndvi and height column have to be scaled.
        @return: dataframe which scaled blue, ndvi and height bands.
        
        """
        pixel_df_copy = pixel_df.copy()

        pixel_df_copy[col_names[0]] = self.scaler_band3.transform(pixel_df_copy[col_names[0]].values.reshape(-1,1))
        pixel_df_copy[col_names[1]] = self.scaler_band5.transform(pixel_df_copy[col_names[1]].values.reshape(-1, 1))
        pixel_df_copy[col_names[2]] = self.scaler_band6.transform(pixel_df_copy[col_names[2]].values.reshape(-1, 1))
        return pixel_df_copy
    
class scaler_class_BH:
    """
    This class is used to scale blue, ndvi and height columns of a pandas dataframe from a .tif file.
    Which should be band 3, band 5 and band 6 respectively.

    Scalers should have been made indepently!
    
    """
    def __init__(self, scaler_file_band3 = "", scaler_file_band6 = "") :
        """
        Init of this class.

        @param scaler_file_band3: Path to a file which contains the scaler for band 3.
        @param scaler_file_band5: Path to a file which contains the scaler for band 5.
        @param scaler_file_band6: Path to a file which contains the scaler for band 6.
        
        """
        self.scaler_band3 = joblib.load(scaler_file_band3)
        self.scaler_band6 = joblib.load(scaler_file_band6)

    def transform(self,pixel_df):
        """
        Transforms the blue, ndvi and height columns of a pandas dataframe.

        @param pixel_df: dataframe in which the blue, ndvi and height column have to be scaled.
        @return: dataframe which scaled blue, ndvi and height bands.
        
        """

        pixel_df['band3'] = self.scaler_band3.transform(pixel_df['band3'].values.reshape(-1,1))        
        pixel_df['band6'] = self.scaler_band6.transform(pixel_df['band6'].values.reshape(-1, 1))
        return pixel_df

class scaler_normalizer_retriever:

    def __init__(self, a_kernel_generator):
            self.kernel_generator =  a_kernel_generator

    def make_scalers_pixel_df(self,parts=1, specific_part=0,  multiprocessing = False, output_name = "",sample = False,):
            """
            
            This function makes a scaler on bands in a .tif file, which can be based on parts of a .tif file instead of the whole file.
            Breaking the .tif file in multiple parts is sometimes used to because the regular file can be too large.

            TODO: Make this a sample part.

            @param parts: The number of parts of which to divide a .tif file into.
            @param specific_part: The specific part to make the scaler on.
            @param multiprocessing: multiprocessing wether to use.
            """

            total_height = self.kernel_generator.get_height() - self.kernel_generator.x_size

            height_parts = round(total_height/parts)
            begin_height = self.kernel_generator.x_size_begin
            end_height = self.kernel_generator.x_size_begin+height_parts

            total_height = self.kernel_generator.get_height()-self.kernel_generator.x_size
            total_width = self.kernel_generator.get_width()-self.kernel_generator.y_size

            height_parts = total_height/parts

            clusters_centers = []
            # Loop through the parts.
            for x_step in tqdm.tqdm(range(specific_part,specific_part+1)):
                print("-------")
                print("Part: "+str(x_step+1)+" of "+str(parts))
                # Calculate the number of permutations for this step.
                permutations = list(itertools.product([x for x in range(begin_height, end_height)], [ y for y in range(self.kernel_generator.y_size_begin, self.kernel_generator.get_width()-self.kernel_generator.y_size_end)]))
                print("Total permutations this step: "+str(len(permutations)))


                print("Retrieving kernels:")
                if multiprocessing == True:
                    p = Pool()
                    pixel_df = p.map(self.get_pixel_multiprocessing,permutations)
                    p.terminate()
                else:
                    pixel_df = [self.get_pixel_multiprocessing(permutation) for permutation in permutations]

                print("Number of pixels:")
                print(len(pixel_df))
                
                pixel_df = [elem for elem in pixel_df if elem is not None]
                                 
                pixel_df = pd.DataFrame(pixel_df, columns= [ "band"+str(band) for band in range(1,len(pixel_df[0])+1)])
                pixel_df = self.make_normalized_scaler(pixel_df, output_name)

            return pixel_df

    def make_normalized_scaler(self, pixel_df,output_name, ahn_scaler = "./scalers/ahn3.save" ):
            """
            Make scalers based on each bands which will be stored in a .save file.

            @param pixel_df: A pandas dataframe with pixels from a .tif file.
            @param output_name: The file name of the save files for each band scaler.
            @oaram ahn_scaler: ahn scaler is used differently so it has to use a different scaler.
            @return pixel_df: a pandas dataframe based on scaled rgb file.
            """

            for band in pixel_df.columns[0:len(pixel_df.columns)-1]:

                band_scaler = MinMaxScaler().fit(pixel_df[band].values.reshape(-1, 1))
                joblib.dump(band_scaler,"./scalers/"+output_name+"_"+str(band)+".save") 
                pixel_df[band] = band_scaler.transform(pixel_df[band].values.reshape(-1, 1))
            
            
            if exists(ahn_scaler):
                 
                    band6_scaler= joblib.load(ahn_scaler) 
            else:
                    band6_scaler = MinMaxScaler().fit(pixel_df['band'+str(len(pixel_df))].values.reshape(-1, 1))               
                    joblib.dump(band6_scaler,ahn_scaler) 

            pixel_df['band'+str(len(pixel_df.columns))] = band6_scaler.transform(pixel_df['band'+str(len(pixel_df.columns))].values.reshape(-1, 1))

            return pixel_df