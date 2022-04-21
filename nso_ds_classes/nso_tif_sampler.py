import nso_ds_classes.nso_tif_kernel
import random
import itertools


class nso_tif_sampler:

    def __init__(self, a_kernel_generator):
        self.kernel_generator =  a_kernel_generator

    def sample_pixels(self,amount = 100):
        height_sample = random.sample(range(0, self.kernel_generator.get_height()), amount)
        width_sample  = random.sample(range(0, self.kernel_generator.get_width()), amount)

        return_samples =[]

        permutations = list(itertools.product([height_sample[x] for x in range(1, len(height_sample))], [width_sample[y] for y in range(1, len(width_sample) )]))
        x_samp =0

        while len(return_samples) < amount:

            
       
            try:
                return_samples.append(self.kernel_generator.get_pixel_value(permutations[x_samp][0], permutations[x_samp][1]))

            except Exception as e:
                if str(e) != "Center pixel is empty":                          
                            print(e)

            x_samp = x_samp+1
            
        return return_samples