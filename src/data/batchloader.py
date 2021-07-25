import pandas as pd
import numpy as np

class BatchLoader:
    def __init__(self, dataset, batch_size, return_labels=None, without_image=False, return_X_y=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_labels = return_labels
        self.data_size = len(dataset)
        self.start = 0
        self.without_image = without_image
        self.return_X_y = return_X_y

    def __iter__(self):
        self.start = 0
        return self

    def __next__(self):
        if self.start >= self.data_size:
            raise StopIteration
        start = self.start
        end = min(start + self.batch_size, self.data_size)
        all_features = []
        image_features = []
        all_labels = []
        if self.without_image:
            featrue_columns = ['Path'] + self.dataset._feature_header.tolist()
            if self.return_labels:
                labels_columns = self.return_labels
            else:
                labels_columns = self.dataset._label_header.tolist()
                
            if self.return_X_y:
                features = self.dataset.df.iloc[start:end][featrue_columns]
                labels = self.dataset.df.iloc[start:end][labels_columns]
                self.start = end
                return features, labels
            else:
                columns = featrue_columns + labels_columns
                data = self.dataset.df.iloc[start:end][columns]
                self.start = end
                return data
        else:
            for i in range(start, end):
                features, image_feature, labels = self.dataset[i]
                all_features.append(features)
                image_features.append(image_feature)
                all_labels.append(labels)
            x_features, x_image = pd.DataFrame(all_features, columns=self.dataset._feature_header), pd.DataFrame(image_features)
            y = pd.DataFrame(all_labels, columns=self.dataset._label_header)
            if self.return_labels:
                if len(self.return_labels) > 1:
                    y = y[self.return_labels]
                else:
                    y = y[self.return_labels[0]]
            self.start = end
            return x_features, x_image, y

