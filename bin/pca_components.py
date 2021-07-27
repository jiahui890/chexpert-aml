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
from sklearn.pipeline import make_pipeline


if __name__ == '__main__':

    default_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=bool, default=False)
    parser.add_argument("--preprocessing", type=str, default="CheXpert-v1.0-small\\config\\default_preprocessing.yaml",
                        help="File path for image preprocessing.")
    parser.add_argument("--file", type=str, default='chexpert')
    parser.add_argument("--map", type=str, default='Random', choices=['U-zero', 'U-one', 'Random'])
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path", type=str, default=default_dir)
    parser.add_argument("--ylabels", nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                                                         'Pleural Effusion'],
                        choices=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                                 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                                 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                                 'Fracture', 'Support Devices'], help="Labels to predict.")

    args = parser.parse_args()
    print(f'==============================================')
    print('Loading dataset')
    base_path = args.path
    preprocessing_path = os.path.join(base_path,args.preprocessing)

    # Alter the directory
    image_path = os.path.join(base_path, "CheXpert-v1.0-small", "CheXpert-v1.0-small")
    train_csv_path = os.path.join(base_path, "CheXpert-v1.0-small", "CheXpert-v1.0-small", "train.csv")
    test_csv_path = os.path.join(base_path, "CheXpert-v1.0-small", "CheXpert-v1.0-small", "valid.csv")
    model_path = os.path.join(base_path, "CheXpert-v1.0-small", "models")
    results_path = os.path.join(base_path, "reports", "figures")

    limit = args.limit
    batch_size = args.batchsize
    return_labels = args.ylabels

    with open(preprocessing_path,'r') as file:
        preprocessing_config = yaml.full_load(file)

    train_dataset = ImageDataset(label_csv_path=train_csv_path, image_path_base=image_path, limit=limit,
                                 transformations=preprocessing_config["transformations"], map_option=args.map)
    #test_dataset = ImageDataset(label_csv_path=test_csv_path, image_path_base=image_path, transformations=preprocessing_config["transformations"])
    print(f'train_dataset: {train_dataset}, {train_csv_path}')
    #print(f'test_dataset: {test_dataset}, {test_csv_path}')
    print(f'==============================================')

    num_batch = train_dataset._num_image // batch_size + bool(train_dataset._num_image // batch_size) #ceiling division

    # =============================================================================

    # Finding optimum PCA components
    if args.pca:
        print(f'Setting up pca')
        pca = IncrementalPCA(whiten=True, batch_size=batch_size)  #set to default values for n_components first
        for i, (x_features, x_image, y) in enumerate(train_dataset.batchloader(batch_size, return_labels)):
            print(f'Training pca on batch {(i + 1)} out of {num_batch}')
            pca.partial_fit(x_image)

        print(f'Saving pca model .sav file... ')
        f_datetime = dt.datetime.now().strftime('%H%M_%d%m%Y')
        pca_fname = os.path.join(model_path, f'{pca}_{batch_size}_{limit}_{"90%"}_{args.map}_{f_datetime}.sav')
        pickle.dump(pca, open(pca_fname, 'wb'))

        # finding optimal
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.9) + 1
        print("Optimal number of PCA Components: ", d)

        # plotting graph
        plt.figure(figsize=(6, 4))
        plt.plot(cumsum, linewidth=3)
        plt.xlabel("Dimensions")
        plt.ylabel("Explained Variance")
        plt.plot([d, d], [0, 0.9], "k:")
        plt.plot([0, d], [0.9, 0.9], "k:")
        plt.plot(d, 0.9, "ko")
        # plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
        #              arrowprops=dict(arrowstyle="->"), fontsize=16)
        plt.grid(True)
        plt.show()

