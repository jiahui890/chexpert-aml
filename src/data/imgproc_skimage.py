import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random


# import image processing library
from src.data import imgproc
from skimage.io import imread as skimread
from skimage.transform import resize as skresize
from skimage.transform import rotate as skrotate
from skimage.transform import rescale as skrescale
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
    def normalize(self, image):
        # convert from integers to floats
        image_norm = image.astype('float32')
        # normalize to range 0 to 1
        image_norm = image_norm / 255.0
        # return normalized images
        return image_norm
    
    def eqhist(self, image):
        return exposure.equalize_hist(image) 

    def adaptive_eqhist(self, image, kernel_size=None, clip_limit=0.03):
        return exposure.equalize_adapthist(image, clip_limit=clip_limit)
    
    def gaussian_blur(self, image, sigma=3, truncate=3):
        return gaussian(image, sigma=sigma, truncate=truncate)
    
    def median_blur(self, image):
        return median(image)
    
    def rotate(self, image, degree=None):
        if degree is None:
            degree = random.randrange(-10,10)
        return skrotate(image, degree)

    def zoom(self, image, percentage=None):
        if percentage is None:
            percentage = random.uniform(1.0, 1.25)
        height, width = image.shape
        enlarge = skresize(image, (round(height*percentage), round(width*percentage)))
        
        ori_height, ori_width = enlarge.shape
        
        diff_height = ori_height-height
        diff_width = ori_width-width
                
        top = diff_height//2
        bottom = top+height
        left = diff_width//2
        right = left+width
        
        return enlarge[top:bottom, left:right]


class TfImageProcessing(imgproc.ImageProcessing):

    def imread(self, path, channels=1):
        image_string = tf.io.read_file(path)
        #Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.io.decode_jpeg(image_string, channels=channels)
        image = tf.image.grayscale_to_rgb(image)
        return image

    def resize(self, image, size=(320, 320)):
        return tf.image.resize(image, [*size])

    def crop(self, image, size=(320, 320)):
        return tf.cast(tf.image.resize_with_crop_or_pad(image, *size), tf.float32)

    # Normalize
    def normalize(self, image):
        # This will convert to float values in [0, 1]
        return image/ 255.0

    def eqhist(self, image):
        return tfa.image.equalize(image)

    def adaptive_eqhist(self, image, kernel_size=None, clip_limit=0.03):
        #No adaptive eqhist available in tensorflow
        return tfa.image.equalize(image)

    def gaussian_blur(self, image, sigma=3):
        return tfa.image.gaussian_filter2d(image, sigma=sigma)

    def median_blur(self, image):
        return tfa.image.median_filter2d(image)

    def rotate(self, image, degree=None):
        if degree is None:
            degree = random.randrange(-10, 10)
        return tfa.image.rotate(image, angles=degree)

    def zoom(self, image, percentage=None, size=(320, 320)):
        if percentage is None:
            percentage = 1.0 / random.uniform(1.0, 1.25)
        image = tf.image.central_crop(image, percentage)
        return tf.image.resize(image, [*size])