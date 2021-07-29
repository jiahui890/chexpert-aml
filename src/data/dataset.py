import numpy as np
import pandas as pd
from src.data.batchloader import BatchLoader
from sklearn.utils.class_weight import compute_class_weight
import os


from src.data.imgproc import get_proc_class


class ImageDataset():
    """ Class for image dataset loading and preprocessing

        Args:
            label_csv_path (string): Path to label csv file.
            image_path_base (string, optional): Base path for the image path in the csv file. Defaults to None.
            proc_module (str, optional): [description]. Defaults to 'skimage'.
            transformations (list, optional): list of image transformations and their arguments. Defaults to [ ('resize', {'size': (320, 320)}), ('flatten', {}) ].
            limit (int, optional): Maxinum limit for loading the dataset. Defaults to None.
    """
    def __init__(self,
                 label_csv_path=None,
                 label_df=None,
                 image_path_base=None,
                 proc_module='skimage',
                 transformations = [
                     ('resize', {'size': (320, 320)}),
                     ('flatten', {})
                 ],
                 frontal_only = False,
                 map_option = None,
                 random_state = 2021,
                 limit = None,
                 clean=True):
        self.image_path_base = image_path_base
        self.imgproc = get_proc_class(proc_module)
        self.transformations = transformations
        self.map_option = map_option
        self.random_state = random_state
        self.limit = limit
        if label_csv_path is not None:
            self.df = pd.read_csv(label_csv_path)
        elif label_df is not None:
            self.df = label_df
        else:
            print('Either label_df or label_csv_path must be specified')
        self._feature_header = self.df.columns[1:5]
        self._label_header = self.df.columns[5::]
        if limit is not None:
            self.df = self.df.sample(n=limit, random_state=self.random_state)
        self.df = self.df.reset_index(drop=True)
        self._num_image = len(self.df)
        self._frontal_only = frontal_only
        if clean:
            self.__clean__()
        if self.map_option is not None:
            self.__map_uncertain__(option=self.map_option)

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
        self.df['Age'] = self.df['Age'] / 100.0
        self.df['Path'] = self.df['Path'].apply(lambda x: x.replace('CheXpert-v1.0-small', self.image_path_base))
        self.df['Frontal/Lateral'] = self.df['Frontal/Lateral'].map({'Frontal': 1, 'Lateral': 0})
        #Frontal view only
        if self._frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 1]
        self.df['Sex'] = self.df['Sex'].map({'Male': 1, 'Female': 0, 'Unknown': 1})
        self.df['AP/PA'] = self.df['AP/PA'].replace(np.nan, 'AP')
        self.df['AP/PA'] = self.df['AP/PA'].map({'AP': 1, 'PA': 0})
        #Not sure why still has np.nan exists
        self.df['AP/PA'] = self.df['AP/PA'].replace(np.nan, 1)
        # Replace np.nan with 0
        self.df[self._label_header] = self.df[self._label_header].replace(np.nan, 0)
        self.df.reset_index(drop=True)
        self._num_image = len(self.df)

    def __map_uncertain__(self, option):
        """"Map the uncertain label of -1 to [0,1] depending on mapping option, replace np.nan with 0.

        Args:
            option : str or dict of column label uncertain approach. Options value can be 'U-zero', 'U-one' and 'Random'
        """
        map_dict = {
            'U-zero':
                {
                    1.0: 1.0,
                    '': 0.0,
                    0.0: 0.0,
                    -1.0: 0.0
                },
            'U-one':
                {
                    1.0: 1.0,
                    '': 0.0,
                    0.0: 0.0,
                    -1.0: 1.0
                }
        }

        op_dict = option

        if type(option) == str:
            op_dict = {col:option for col in self._label_header} 

        columns = list(op_dict.keys())

        for col, opt in op_dict.items():
            if opt == 'U-zero' or opt == 'U-one':
                self.df[col] = self.df[col].map(map_dict[opt])
            elif opt == 'Random':
                sum_positive = (self.df[col] == 1.0).sum()
                sum_negative = (self.df[col] == 0.0).sum()
                prob_positive = sum_positive / (sum_positive + sum_negative)
                list_size = self.df[self.df[col] == -1.0][col].size
                
                random_list = np.random.choice(a=[0, 1], p=[1 - prob_positive, prob_positive], size=list_size)
                self.df.loc[self.df[col] == -1.0, col] = random_list

    def split(self, validsize, transformations=None):
        self.valid_df = self.df.sample(n=round(validsize*self.df.shape[0]), random_state=self.random_state)
        self.df = (self.df.drop(self.valid_df.index)
                          .reset_index(drop=True))
        self._num_image = len(self.df)
        self.valid_df = self.valid_df.reset_index(drop=True)
        if transformations is None:
            transformations = self.transformations

        return ImageDataset(label_df=self.valid_df, image_path_base=self.image_path_base,
                            transformations=transformations, map_option=self.map_option,
                            frontal_only=self._frontal_only, clean=False)

    def batchloader(self, batch_size, return_labels=None, without_image=False, return_X_y=True):
        """Loader for loading dataset in batch.

        Args:
            batch_size (int): Size for one batch.
            return_labels ([list, optional): List of labels to be return. If value is None it will return all labels. Defaults to None.

        Returns:
            BatchLoader: a literator.
        """
        return BatchLoader(self, batch_size, return_labels, without_image=without_image, return_X_y=return_X_y)

    def load(self, return_labels=None, without_image=False, return_X_y=True):
        """Load the entire dataset.

        Args:
            return_labels ([list, optional): List of labels to be return. If value is None it will return all labels. Defaults to None.

        Returns:
            X, y: Pandas DataFrame
        """
        return next(iter(BatchLoader(self, self._num_image, return_labels, without_image=without_image, return_X_y=return_X_y)))

    def get_class_weights(self, return_labels=None):
        """"Get the class weights.

        Args:
            return_labels ([list, optional): List of labels to be return. If value is None it will return all labels. Defaults to None.

        Returns:
            class_weight_list: List of class weight dict
        """
        class_weight_list = np.zeros(shape=(len(return_labels), 2))
        for idx, col in enumerate(return_labels):
            y_data = self.df[col].values.flatten()
            class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_data)
            #class_weight_list[idx] = {0: class_weight[0], 1: class_weight[1]}
            class_weight_list[idx, 0] = class_weight[0]
            class_weight_list[idx, 1] = class_weight[1]

        return class_weight_list


