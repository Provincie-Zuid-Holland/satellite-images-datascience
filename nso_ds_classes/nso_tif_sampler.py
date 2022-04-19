import nso_ds_classes.nso_tif_kernel
import random


class nso_tif_sampler:

    def __init__(self, a_kernel_generator):
        self.kernel_generator =  a_kernel_generator

    def sample_pixels(self,amount = 100):
        height_sample = random.sample(range(0, self.kernel_generator.get_height()), amount)
        width_sample  = random.sample(range(0, self.kernel_generator.get_width()), amount)

        return_samples =[]

        for x_samp in range(0, len(height_sample)):

            try:
                return_samples.append(self.kernel_generator.get_pixel_value(height_sample[x_samp], width_sample[x_samp]))
            except:
                print("Empty value")

        return return_samples