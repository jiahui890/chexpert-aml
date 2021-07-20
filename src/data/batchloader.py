import pandas as pd
import numpy as np

class BatchLoader:
    def __init__(self, dataset, batch_size, return_labels=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_labels = return_labels
        self.data_size = len(dataset)
        self.start = 0

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
        for i in range(start, end):
            features, image_feature, labels = self.dataset[i]
            all_features.append(features)
            image_features.append(image_feature)
            all_labels.append(labels)
        x_features, x_image = pd.DataFrame(all_features), pd.DataFrame(image_features)
        y = pd.DataFrame(all_labels, columns=self.dataset._label_header)
        if self.return_labels:
            y = y[self.return_labels]
        self.start = end
        return x_features, x_image, y

