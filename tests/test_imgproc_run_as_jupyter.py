#%%
import os

if(os.getcwd().endswith('tests')):
    os.chdir('..')
print(os.getcwd())

from src.data import imgproc
# from notebooks import preprocessing_config
import matplotlib.pyplot as plt
import yaml
#%%
with open('notebooks/preprocessing_config.yaml', 'r') as file:
    preprocessing_config = yaml.full_load(file)

path = "./tests/view1_frontal.jpg"
proc_class = imgproc.get_proc_class('skimage')
image = proc_class.imread(path)
result = proc_class.transform(image, preprocessing_config['transformations'])
plt.imshow(result, cmap=plt.cm.gray)

# %%
