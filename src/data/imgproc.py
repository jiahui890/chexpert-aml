from abc import ABC, abstractmethod
import numpy as np

class ImageProcessing(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def imread(self, path):
        pass

    @abstractmethod
    def resize(self, path, size):
        pass

    def flatten(self, image):
        flatted = np.ndarray.flatten(image)
        return flatted

    def transform(self, image, transformations): 
        for trans, args in transformations:
            image = getattr(self, trans)(image, **args)
        return image

def get_proc_class(module_name):
    if module_name == 'skimage':
        from .imgproc_skimage import SKImageProcessing
        return SKImageProcessing()
    else:
        raise Exception(f'Unkown module name {module_name} for image process class')

