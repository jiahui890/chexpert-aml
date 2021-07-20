import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.data.batchloader import BatchLoader
import os


from src.data.imgproc import get_proc_class


class ImageDataset(Dataset):
    """ Class for image dataset loading and preprocessing

        Args:
            label_csv_path (string): Path to label csv file.
            image_path_base (string, optional): Base path for the image path in the csv file. Defaults to None.
            proc_module (str, optional): [description]. Defaults to 'skimage'.
            transformations (list, optional): list of image transformations and their arguments. Defaults to [ ('resize', {'size': (320, 320)}), ('flatten', {}) ].
            limit (int, optional): Maxinum limit for loading the dataset. Defaults to None.
    """
    # TODO: make Uncertainty Approaches cofigurable
    # TODO: return all the lables, now only [Cardiomegaly  Edema  Consolidation  Atelectasis  Pleural Effusion]
    def __init__(self,
                 label_csv_path,
                 image_path_base=None,
                 proc_module='skimage',
                 transformations = [
                     ('resize', {'size': (320, 320)}),
                     ('flatten', {})
                 ],
                 map_option = None,
                 limit = None):
        self.image_path_base = image_path_base
        self.imgproc = get_proc_class(proc_module)
        self.transformations = transformations
        self.limit = limit
        self.df = pd.read_csv(label_csv_path)
        self._feature_header = self.df.columns[1:5]
        self._label_header = self.df.columns[5::]
        if limit is not None:
            self.df = self.df.sample(n=limit)
        self.df.reset_index(drop=True)
        self._num_image = len(self.df)
        self.__clean__()
        if map_option is not None:
            self.__map_uncertain__(option=map_option)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = self.imgproc.imread(self.df['Path'].iloc[idx])
        transformed = self.imgproc.transform(image, self.transformations)
        features = self.df[self._feature_header].iloc[idx].values
        labels = self.df[self._label_header ].iloc[idx].values
        return (features, transformed, labels)

    def __clean__(self):
        """"Perform basic data cleaning
        """
        self.df['Path'] = self.df['Path'].apply(lambda x: x.replace('CheXpert-v1.0-small', self.image_path_base))
        self.df['Frontal/Lateral'] = self.df['Frontal/Lateral'].map({'Frontal': 1, 'Lateral': 0})
        self.df['Sex'] = self.df['Sex'].map({'Male': 1, 'Female': 0, 'Unknown': 1})
        self.df['AP/PA'] = self.df['AP/PA'].replace(np.nan, 'AP')
        self.df['AP/PA'] = self.df['AP/PA'].map({'AP': 1, 'PA': 0})
        #Not sure why still has np.nan exists
        self.df['AP/PA'] = self.df['AP/PA'].replace(np.nan, 1)
        self.df.reset_index(drop=True)
        self._num_image = len(self.df)

    def __map_uncertain__(self, option=None, columns=None):
        """"Map the uncertain label of -1 to [0,1] depending on mapping option, replace np.nan with 0.

        Args:
            option : List of options 'U-zero', 'U-one' and 'Random'
            columns: List of columns to map
        """
        map_dict = {
            'U-zero':
                {
                    1.0: 1,
                    '': 0,
                    0.0: 0,
                    -1.0: 0
                },
            'U-one':
                {
                    1.0: 1,
                    '': 0,
                    0.0: 0,
                    -1.0: 1
                }
        }

        # Replace np.nan with 0
        if columns == None:
            columns = self._label_header
        self.df[columns] = self.df[columns].replace(np.nan, 0)

        if option == 'U-zero' or option == 'U-one':
            for col in columns:
                self.df[col] = self.df[col].map(map_dict[option])
        elif option == 'Random':
            for col in columns:
                sum_positive = (self.df[col] == 1.0).sum()
                sum_negative = (self.df[col] == 0.0).sum()
                prob_positive = sum_positive / (sum_positive + sum_negative)
                list_size = self.df[self.df[col] == -1.0][col].size
                random_list = np.random.choice(a=[0, 1], p=[1 - prob_positive, prob_positive], size=list_size)
                self.df.loc[self.df[col] == -1.0, col] = random_list
    
    def batchloader(self, batch_size, return_labels=None):
        """Loader for loading dataset in batch.

        Args:
            batch_size (int): Size for one batch.
            return_labels ([list, optional): List of labels to be return. If value is None it will return all labels. Defaults to None.

        Returns:
            BatchLoader: a literator.
        """
        return BatchLoader(self, batch_size, return_labels)

    def load(self, return_labels=None):
        """Load the entire dataset.

        Args:
            return_labels ([list, optional): List of labels to be return. If value is None it will return all labels. Defaults to None.

        Returns:
            X, y: Pandas DataFrame
        """
        return next(iter(BatchLoader(self, self._num_image, return_labels)))


