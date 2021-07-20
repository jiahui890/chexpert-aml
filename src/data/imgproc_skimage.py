from src.data import imgproc
import numpy as np


# import image processing library
from skimage.io import imread as skimread
from skimage.transform import resize as skresize
# from skimage.color import rgb2grey
from skimage.filters import gaussian, laplace, median
from skimage import exposure

class SKImageProcessing(imgproc.ImageProcessing):

    def imread(self, path):
        return skimread(path)

    def resize(self, image, size=(320, 320)):
        return skresize(image, size)
    
    def crop(self, image, size=(320,320)):
        ori_height, ori_width = image.shape
        height = size[0]
        width = size[1]
        
        diff_height = ori_height-height
        diff_width = ori_width-width
                
        top = diff_height//2
        bottom = top+height
        left = diff_width//2
        right = left+width
        
        return image[top:bottom, left:right]

    # Normalize
    def normalize(image):
        # convert from integers to floats
        image_norm = image.astype('float32')
        # normalize to range -1 to 1
        image_norm = image_norm / 127.5 -1
        # return normalized images
        return image_norm
    
    def eqhist(self, image):
        return exposure.equalize_hist(image) 
    
    def adaptive_eqhist(self, image):
        return exposure.equalize_adapthist(image, clip_limit=0.03)
    
    def gaussian_blur(self, image):
        return gaussian(image, sigma=3, truncate=3)
    
    def median_blur(self, image):
        return median(image)
    

