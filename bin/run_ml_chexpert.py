import os
import sys
sys.path.append('..')
import argparse
import datetime as dt
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
from src.data.dataset import ImageDataset
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import logging
from datetime import datetime

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    #TODO: Add argparse for image transformation
    parser.add_argument("--pca", type=bool, default=False)
    parser.add_argument("--preprocessing", type=str, default="config/default_preprocessing.yaml")
    parser.add_argument("--file", type=str, default='chexpert')
    parser.add_argument("--map", type=str, default='Random', choices=['U-zero', 'U-one', 'Random'])
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--validsize", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path", type=str, default=default_dir)
    parser.add_argument("--ylabels", nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                        'Pleural Effusion'])

    args = parser.parse_args()
    logger.info(f'==============================================')
    logger.info('Loading dataset')
    base_path = args.path
    preprocessing_path = os.path.join(base_path,args.preprocessing)
    image_path = os.path.join(base_path, "data", "raw")
    train_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "train.csv")
    test_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "valid.csv")
    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "reports", "figures")
    limit = args.limit
    batch_size = args.batchsize
    return_labels = args.ylabels

    with open(preprocessing_path,'r') as file:
        preprocessing_config = yaml.full_load(file)

    train_dataset = ImageDataset(label_csv_path=train_csv_path, image_path_base=image_path, limit=limit,
                                 transformations=preprocessing_config["transformations"], map_option=args.map)
    if args.validsize is not None:
        valid_dataset = train_dataset.split(validsize=args.validsize)
    test_dataset = ImageDataset(label_csv_path=test_csv_path, image_path_base=image_path, transformations=preprocessing_config["transformations"])
    logger.info(f'train_dataset: {train_dataset}, {train_csv_path}')
    logger.info(f'test_dataset: {test_dataset}, {test_csv_path}')
    logger.info(f'==============================================')

    num_batch = train_dataset._num_image // batch_size + bool(train_dataset._num_image % batch_size) #ceiling division

    # TODO: Setup your own model here
    if args.pca:
        start_time = datetime.now()
        logger.info(f'Setting up pca')
        pca = IncrementalPCA(n_components=50, whiten=True, batch_size=batch_size)
        for i, (x_features, x_image, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
            logger.info(f'Training pca on batch {(i + 1)} out of {num_batch}')
            pca.partial_fit(x_image)
        end_time = datetime.now()
        logger.info('PCA fit duration: {}'.format(end_time - start_time))

        logger.info(f'Saving pca model .sav file... ')
        f_datetime = dt.datetime.now().strftime('%H%M_%d%m%Y')
        pca_fname = os.path.join(model_path, f'{pca}_{batch_size}_{args.map}_{f_datetime}.sav')
        pickle.dump(pca, open(pca_fname, 'wb'))

    base_model = MultinomialNB()
    logger.info(f'model: {base_model}')

    if len(return_labels) > 1:
        model = MultiOutputClassifier(base_model)
        logger.info ("Extending to Multi Output Classifer")
    else:
        model = base_model

    classes = np.array([[0, 1] for y in return_labels]).astype(np.float32)
    start_time = datetime.now()
    for i, (x_features, x_image, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
        if args.pca:
            x_image = MinMaxScaler().fit_transform(pca.transform(x_image))
        X = pd.concat([pd.DataFrame(x_features), pd.DataFrame(x_image)], axis=1)
        logger.info(f'Training model on batch {(i + 1)} out of {num_batch}')
        model.partial_fit(X, y, classes=classes)
    end_time = datetime.now()
    logger.info('Model training duration: {}'.format(end_time - start_time))

    logger.info(f'Saving model .sav file... ')
    f_datetime = dt.datetime.now().strftime('%H%M_%d%m%Y')
    model_fname = os.path.join(model_path, f'{args.file}_{batch_size}_{args.map}_{f_datetime}.sav')
    pickle.dump(model, open(model_fname, 'wb'))
    logger.info(f'Running model on test dataset...')
    x_features_test, x_image_test, y_test_multi = test_dataset.load(return_labels)
    if args.pca:
        x_image_test = MinMaxScaler().fit_transform(pca.transform(x_image_test))
    X_test = pd.concat([pd.DataFrame(x_features_test), pd.DataFrame(x_image_test)], axis=1)
    y_pred_multi = np.array(model.predict_proba(X_test))
    y_pred_labels = np.array(model.predict(X_test))
    logger.info(f'==============================================')
    logger.info(f'Verification results')
    logger.info(f'==============================================\n')

    for idx, label in enumerate(return_labels):
        y_test = y_test_multi[label]
        y_pred = y_pred_multi[idx, :, 1]
        y_pred_label = y_pred_labels[:, idx]
        auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_label)
        f1 = f1_score(y_true=y_test, y_pred=y_pred_label)
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
        ax.plot(fpr, tpr, label=f"AUC:{auc}")
        ax.legend()
        ax.set_title(f"ROC curve for {label}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig_fname = os.path.join(results_path, f"{args.file}_{batch_size}_{args.map}_{label}_{f_datetime}.png")
        text_fname = os.path.join(results_path, f"{args.file}_{batch_size}_{args.map}_{label}_{f_datetime}.txt")
        f = open(text_fname, "w")
        fig.savefig(fig_fname, dpi=72)
        fig.clear()
        if args.pca:
            logger.info(f'PCA components: {pca.n_components}')
            f.write(f'PCA components: {pca.n_components}\n')
        logger.info(f'{label}')
        logger.info(f'roc_auc_score: {auc}')
        logger.info(f'accuracy: {accuracy}')
        logger.info(f'f1-score: {f1}')
        f.write(f'{label}\n')
        f.write(f'roc_auc_score: {auc}\n')
        f.write(f'accuracy: {accuracy}\n')
        f.write(f'f1-score: {f1}\n')
