from src.data import imgproc
import numpy as np


# import image processing library
from skimage.io import imread as skimread
from skimage.transform import resize as skresize
# from skimage.color import rgb2grey

class SKImageProcessing(imgproc.ImageProcessing):

    def imread(self, path):
        return skimread(path)

    def resize(self, image, size=(320, 320)):
        return skresize(image, size)


