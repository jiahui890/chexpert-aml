import os
import sys

from pandas.core.indexes import base

sys.path.append('..')
import argparse
import datetime as dt
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from src.data.imgproc import tf_read_image
from src.data.dataset import ImageDataset
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
from src.models.sklearn_models import models, param_grids
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import logging
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__file__)


def parse_args():
    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pca_pretrained",
        type=str,
        default='IncrementalPCA_128_500_Random_0350_26072021.sav',
        help=".sav file path for pretrained pca model.")
    parser.add_argument("--preprocessing",
                        type=str,
                        default="config/default_preprocessing.yaml",
                        help="File path for image preprocessing.")
    parser.add_argument("--file",
                        type=str,
                        default='chexpert',
                        help="Filename prefix. "
                        "You should give a meaningful name for easy tracking.")
    parser.add_argument("--map",
                        type=str,
                        default='Random',
                        choices=['U-zero', 'U-one', 'Random'],
                        help="Option for mapping uncertain labels.")
    parser.add_argument("--limit",
                        type=int,
                        default=5000,
                        help="Maximum dataset size capped.")
    parser.add_argument("--path",
                        type=str,
                        default=default_dir,
                        help="Base path.")
    parser.add_argument("--ylabels",
                        nargs='+',
                        default=[
                            'Atelectasis', 'Cardiomegaly', 'Consolidation',
                            'Edema', 'Pleural Effusion'
                        ],
                        choices=[
                            'No Finding', 'Enlarged Cardiomediastinum',
                            'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                            'Edema', 'Consolidation', 'Pneumonia',
                            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other', 'Fracture', 'Support Devices'
                        ],
                        help="Labels to predict.")
    parser.add_argument("--model",
                        type=str,
                        default='SGDClassifier',
                        choices=[m for m in models],
                        help="Choice of model.")
    parser.add_argument("--n_jobs",
                        type=int,
                        default=-1,
                        help="Number of cores for multi-processing.")

    args = parser.parse_args()
    return args


def load_pca(args):
    try:
        pca_f_path = os.path.join(args.path, "models", args.pca_pretrained)
        with open(pca_f_path, 'rb') as file:
            pca = pickle.load(file)
        logger.info(f'Pretrained pca {pca_f_path} .sav file loaded.')
        logger.info(f'Pretrained pca: {pca}')
    except:
        logger.error(
            f'Pretrained pca {pca_f_path} .sav file cannot be loaded!')
    return pca


def load_dataset(args):
    base_path = args.path
    preprocessing_path = os.path.join(base_path, args.preprocessing)
    image_path = os.path.join(base_path, "data", "raw")
    train_csv_path = os.path.join(base_path, "data", "raw",
                                  "CheXpert-v1.0-small", "train.csv")

    with open(preprocessing_path, 'r') as file:
        preprocessing_config = yaml.full_load(file)

    train_dataset = ImageDataset(
        label_csv_path=train_csv_path,
        image_path_base=image_path,
        limit=args.limit,
        transformations=preprocessing_config["transformations"],
        map_option=args.map)

    return train_dataset


def search_cv(model, param_grid, X, y, scoring, n_jobs):
    grid_search = GridSearchCV(model,
                               param_grid,
                               cv=5,
                               scoring=scoring,
                               return_train_score=True,
                               n_jobs=n_jobs)
    grid_search.fit(X, y)
    return grid_search


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    args = parse_args()
    logger.info(f'==============================================')
    logger.info('Loading dataset')
    train_dataset = load_dataset(args)

    limit = args.limit
    return_labels = args.ylabels
    logger.info(f'==============================================')
    logger.info(f'Running search cv on data size limit: {limit}')
    logger.info(f'Labels to predict: {return_labels}')

    pca = None
    if args.pca_pretrained:
        pca = load_pca(args)

    base_model = models[args.model]
    logger.info(f'model: {base_model}')
    param_grid = param_grids[args.model]
    logger.info(f'param_grid: {param_grid}')

    x_features_train, x_image_train, y_train_multi = train_dataset.load(
        return_labels)
    if pca:
        x_image_train = MinMaxScaler().fit_transform(
            pca.transform(x_image_train))

    X_train = pd.concat(
        [pd.DataFrame(x_features_train),
         pd.DataFrame(x_image_train)], axis=1)

    for label in return_labels:
        y_train = y_train_multi[label]
        logger.info(f'==============================================')
        logger.info(f'Run search_cv for label: {label}')
        scoring = 'roc_auc'
        search = search_cv(base_model,
                           param_grid,
                           X_train,
                           y_train,
                           scoring,
                           n_jobs=args.n_jobs)
        logger.info(
            f'Best score for label {label} is: {search.best_score_}'
        )
        logger.info(
            f'Best params for label {label} are: {search.best_params_}'
        )
        # cvres = pd.DataFrame(search.cv_results_)
        # print(cvres)


if __name__ == '__main__':
    main()