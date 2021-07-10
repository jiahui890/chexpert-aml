# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
# system libraries
import os
import warnings

from torch.utils import data
  
# ignoring all the warnings
warnings.simplefilter('ignore')
  
# import data handling libraries
import numpy as np
import pandas as pd
  
# importing data visualisation libraires
# import matplotlib.pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')
  
# %%
from src.data.dataset import ImageDataset

# %%
print('create dataset')
base_path = os.path.expanduser("~/Downloads")
train_csv_path = os.path.join(base_path, "CheXpert-v1.0-small/train.csv")
test_csv_path = os.path.join(base_path, "CheXpert-v1.0-small/valid.csv")
limit = 100
batch_size = 10
return_labels = ["Cardiomegaly"]
train_dataset = ImageDataset(label_csv_path=train_csv_path, image_path_base=base_path, limit=limit)
test_dataset = ImageDataset(label_csv_path=test_csv_path, image_path_base=base_path, limit=10)

print(f'train_dataset: {train_dataset}')
print(f'test_dataset: {test_dataset}')


# %%
from sklearn.decomposition import PCA                   
from sklearn.svm import SVC     
from sklearn.linear_model import SGDClassifier                      
from sklearn.pipeline import make_pipeline                    
from sklearn import metrics                           

# TODO: try sklearn.decomposition.IncrementalPCA 
pca = PCA(n_components = 150, whiten = True, random_state = 0)
svc = SVC(kernel ='rbf', class_weight ='balanced')
lr = SGDClassifier()
# model = make_pipeline(pca, svc)
model = lr
print(f'model: {model}')

# %%
classes = np.array([0, 1]).astype(np.float32)
for i, (X, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
    print(f'partial_fix batch {i+1}')
    model.partial_fit(X, y, classes=classes)

# %%
print(f'test model')
X_test, y_test = test_dataset.load(return_labels)
y_pred = model.predict(X)
print(f'y_test: {y_test}')
print(f'y_pred: {y_pred}')

      
# Finally, we test our roc_auc_score in using the following code:
auc = metrics.roc_auc_score(y_test, y_pred)
print(f'roc_auc_score: {auc}')
