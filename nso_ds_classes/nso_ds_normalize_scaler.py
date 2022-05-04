
import joblib

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

    def transform(self,pixel_df):
        """
        Transforms the blue, ndvi and height columns of a pandas dataframe.

        @param pixel_df: dataframe in which the blue, ndvi and height column have to be scaled.
        @return: dataframe which scaled blue, ndvi and height bands.
        
        """

        pixel_df['band3'] = self.scaler_band3.transform(pixel_df['band3'].values.reshape(-1,1))
        pixel_df['band5'] = self.scaler_band5.transform(pixel_df['band5'].values.reshape(-1, 1))
        pixel_df['band6'] = self.scaler_band6.transform(pixel_df['band6'].values.reshape(-1, 1))
        return pixel_df