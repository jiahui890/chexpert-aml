import os
import warnings
import argparse
import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
from src.data.dataset import ImageDataset
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, roc_curve

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #TODO: Add argparse for image transformation
    parser.add_argument("-f", "--file", type=str, default='chexpert')
    parser.add_argument("-m", "--map", type=str, default='Random', choices=['U-zero', 'U-one', 'Random'])
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("-p", "--path", type=str, default=r"C:\Users\songh\Google Drive\ISS610\chexpert-aml")
    parser.add_argument("-y", "--ylabels", nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                        'Pleural Effusion'])

    args = parser.parse_args()
    print(f'==============================================')
    print('Loading dataset')
    base_path = args.path
    image_path = os.path.join(base_path, "data", "raw")
    train_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "train.csv")
    test_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "valid.csv")
    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "reports", "figures")
    limit = args.limit
    batch_size = args.batchsize
    return_labels = args.ylabels
    train_dataset = ImageDataset(label_csv_path=train_csv_path, image_path_base=image_path, limit=limit)
    train_dataset.clean()
    test_dataset = ImageDataset(label_csv_path=test_csv_path, image_path_base=image_path)
    test_dataset.clean()
    train_dataset.map_uncertain(option=args.map)

    print(f'train_dataset: {train_dataset}, {train_csv_path}')
    print(f'test_dataset: {test_dataset}, {test_csv_path}')
    print(f'==============================================')

    # TODO: Setup your own model here
    #pca = IncrementalPCA(n_components=50, whiten=True)
    base_model = MultinomialNB()
    print(f'model: {base_model}')

    if len(return_labels) > 1:
        model = MultiOutputClassifier(base_model)
        print ("Extending to Multi Output Classifer")
    else:
        model = base_model

    num_batch = train_dataset._num_image // batch_size

    classes = np.array([[0, 1] for y in return_labels]).astype(np.float32)
    for i, (X, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
        print(f'Training model on batch {(i + 1)} out of {num_batch}')
        model.partial_fit(X, y, classes=classes)
    print(f'Saving model .sav file... ')
    model_fname = os.path.join(model_path, f'{args.file}_{dt.datetime.now().strftime("%H%M_%d%m%Y")}.sav')
    pickle.dump(model, open(model_fname, 'wb'))
    print(f'Running model on test dataset...')
    X_test, y_test_multi = test_dataset.load(return_labels)
    y_pred_multi = np.array(model.predict_proba(X_test))
    print(f'==============================================')
    print(f'Verification results')
    print(f'==============================================\n')
    for idx, label in enumerate(return_labels):
        y_test = y_test_multi[label]
        y_pred = y_pred_multi[idx, :, 1]
        auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
        ax.plot(fpr, tpr, label=f"AUC:{auc}")
        ax.legend()
        ax.set_title(f"ROC curve for {label}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig_fname = os.path.join(results_path, f"{args.file}_{label}_{dt.datetime.now().strftime('%H%M_%d%m%Y')}.png")
        fig.savefig(fig_fname, dpi=72)
        fig.clear()
        print (f'{label}')
        print(f'roc_auc_score: {auc}')
