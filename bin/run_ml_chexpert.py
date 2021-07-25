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
from src.data.dataset import ImageDataset
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
from src.models.sklearn_models import models
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
    parser.add_argument("--pca", type=str, default='False', choices=['True', 'False'], help="Option to train pca model.")
    parser.add_argument("--pca_pretrained", type=str, default=None, help=".sav file path for pretrained pca model.")
    parser.add_argument("--pca_n_components", type=int, default=32, help="n_components for pca.")
    parser.add_argument("--preprocessing", type=str, default="config/default_preprocessing.yaml",
                        help="File path for image preprocessing.")
    parser.add_argument("--file", type=str, default='chexpert', help="Filename prefix. "
                                                                     "You should give a meaningful name for easy tracking.")
    parser.add_argument("--map", type=str, default='Random', choices=['U-zero', 'U-one', 'Random'],
                        help="Option for mapping uncertain labels.")
    parser.add_argument("--batchsize", type=int, default=32, help="Training batch size.")
    parser.add_argument("--validsize", type=float, default=None, help="Validation dataset size.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum dataset size capped.")
    parser.add_argument("--path", type=str, default=default_dir, help="Base path.")
    parser.add_argument("--ylabels", nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                        'Pleural Effusion'], choices=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                        'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                        'Fracture', 'Support Devices'], help="Labels to predict.")
    parser.add_argument("--model", type=str, default='MultinomialNB', choices=[m for m in models], help="Choice of model.")
    parser.add_argument("--model_pretrained", type=str, default=None,
                        help=".sav file for pretrained classifer e.g NaiveBayes_50_50_Random_1259_25072021.sav")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores for multi-processing.")

    args = parser.parse_args()
    logger.info(f'==============================================')
    logger.info('Loading dataset')
    process_pca = False
    base_path = args.path
    preprocessing_path = os.path.join(base_path, args.preprocessing)
    image_path = os.path.join(base_path, "data", "raw")
    train_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "train.csv")
    test_csv_path = os.path.join(base_path, "data", "raw", "CheXpert-v1.0-small", "valid.csv")
    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "reports", "figures")
    limit = args.limit
    batch_size = args.batchsize
    return_labels = args.ylabels
    logger.info(f'==============================================')
    logger.info(f'Training data size limit: {limit}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Labels to predict: {return_labels}')

    if args.pca == 'True':
        process_pca = True
    elif args.pca == 'False':
        process_pca = False

    if args.pca_n_components > batch_size:
        raise ValueError(f'Number of pca components {args.pca_n_component} is larger than batch size {batch_size}!')

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

    if process_pca:
        start_time = datetime.now()
        logger.info(f'Setting up pca')
        if args.pca_pretrained is None:
            pca = IncrementalPCA(n_components=args.pca_n_components, whiten=True, batch_size=batch_size)
            for i, (x_features, x_image, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
                logger.info(f'Training pca on batch {(i + 1)} out of {num_batch}')
                pca.partial_fit(x_image)
            end_time = datetime.now()
            logger.info('PCA fit duration: {}'.format(end_time - start_time))
            logger.info(f'Saving pca model .sav file... ')
            f_datetime = dt.datetime.now().strftime('%H%M_%d%m%Y')
            pca_fname = os.path.join(model_path,
                                     f'{pca.__class__.__name__}_{args.pca_n_components}_{batch_size}_{args.map}_{f_datetime}.sav')
            pickle.dump(pca, open(pca_fname, 'wb'))
        else:
            try:
                pca_f_path = os.path.join(base_path, "models", args.pca_pretrained)
                with open(pca_f_path, 'rb') as file:
                    pca = pickle.load(file)
                logger.info(f'Pretrained pca {pca_f_path} .sav file loaded.')
                logger.info(f'Pretrained pca: {pca}')
            except:
                logger.error(f'Pretrained pca {pca_f_path} .sav file cannot be loaded!')

    base_model = models[args.model]
    logger.info(f'model: {base_model}')

    if len(return_labels) > 1:
        model = MultiOutputClassifier(base_model, n_jobs=args.n_jobs)
        logger.info ("Extending to Multi Output Classifer")
    else:
        model = base_model

    classes = np.array([[0, 1] for y in return_labels]).astype(np.float32)
    start_time = datetime.now()
    f_datetime = start_time.strftime('%H%M_%d%m%Y')
    f_date_dir = start_time.strftime('%d%m%Y')

    if args.model_pretrained is None:
        for i, (x_features, x_image, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
            if process_pca:
                x_image = MinMaxScaler().fit_transform(pca.transform(x_image))
            X = pd.concat([pd.DataFrame(x_features), pd.DataFrame(x_image)], axis=1)
            if num_batch == 1:
                logger.info(f'Training model full batch without partial fit')
                model.fit(X, y)
            else:
                logger.info(f'Training model on batch {(i + 1)} out of {num_batch}')
                model.partial_fit(X, y, classes=classes)
        end_time = datetime.now()
        logger.info('Model training duration: {}'.format(end_time - start_time))
        logger.info(f'Saving model .sav file... ')
        model_fname = os.path.join(model_path, f'{args.file}_{batch_size}_{args.map}_{f_datetime}.sav')
        pickle.dump(model, open(model_fname, 'wb'))
    else:
        try:
            model_f_path = os.path.join(base_path, "models", args.model_pretrained)
            with open(model_f_path, 'rb') as file:
                model = pickle.load(file)
            logger.info(f'Pretrained model {model_f_path} .sav file loaded.')
            logger.info(f'Pretrained model: {model}')
        except:
            logger.error(f'Pretrained model {model_f_path} .sav file cannot be loaded!')

    logger.info(f'Running model on test dataset...')
    x_features_test, x_image_test, y_test_multi = test_dataset.load(return_labels)
    if process_pca:
        x_image_test = MinMaxScaler().fit_transform(pca.transform(x_image_test))
    X_test = pd.concat([pd.DataFrame(x_features_test), pd.DataFrame(x_image_test)], axis=1)

    y_pred_multi = np.array(model.predict_proba(X_test))
    y_pred_labels = np.array(model.predict(X_test))
    logger.info(f'*********************************************')
    logger.info(f'         Verification results                ')
    logger.info(f'*********************************************\n')

    if process_pca:
        model_title = f"(PCA {pca.n_components} {args.model} | {args.map})"
    else:
        model_title = f"({args.model} | {args.map})"

    results_path = os.path.join(results_path, f_date_dir, args.file, args.model)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

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
        ax.plot(fpr, tpr, label=f"AUC: %.3f " %(auc))
        ax.legend()
        ax.set_title(f"ROC curve for {label} {model_title}", fontsize=8)
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        fig_fname = os.path.join(results_path, f"{args.file}_{args.model}_{batch_size}_{args.map}_{label}_{f_datetime}.png")
        text_fname = os.path.join(results_path, f"{args.file}_{args.model}_{batch_size}_{args.map}_{label}_{f_datetime}.txt")
        f = open(text_fname, "w")
        fig.savefig(fig_fname, dpi=200)
        fig.clear()

        logger.info(f'========================================')
        logger.info(f'{label}')
        logger.info(f'========================================')

        f.write(f'========================================\n')
        f.write(f'{label}\n')
        f.write(f'========================================\n')

        if process_pca:
            logger.info(f'PCA components: {pca.n_components}')
            f.write(f'PCA components: {pca.n_components}\n')
        logger.info(f'Model: {model}')
        logger.info(f'Uncertain label mapping: {args.map}')
        logger.info(f'Image transformation: {preprocessing_config["transformations"]}')
        logger.info(f'roc_auc_score: {auc}')
        logger.info(f'accuracy: {accuracy}')
        logger.info(f'f1-score: {f1}')

        f.write(f'Model: {model}\n')
        f.write(f'Uncertain label mapping: {args.map}\n')
        f.write(f'Image transformation: {preprocessing_config["transformations"]}\n')
        f.write(f'roc_auc_score: {auc}\n')
        f.write(f'accuracy: {accuracy}\n')
        f.write(f'f1-score: {f1}\n')
