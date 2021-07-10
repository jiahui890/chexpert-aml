import numpy as np
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
                 limit = None):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self.imgproc = get_proc_class(proc_module)
        self.transformations = transformations
        self.limit = limit
        # Uncertainty Approaches
        self.dict = [
            # U-Zero
            {
                '1.0': '1',
                '': '0',
                '0.0': '0',
                '-1.0': '0'
            },
            # U-One
            {
                '1.0': '1',
                '': '0',
                '0.0': '0',
                '-1.0': '1'
            },
        ]
        with open(label_csv_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7], header[10], header[11], header[13], header[15]
            ]
            line_count = 0
            for line in f:
                if self.limit and line_count >= self.limit:
                    break
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(image_path_base, image_path)
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                line_count += 1
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = self.imgproc.imread(self._image_paths[idx])
        transformed = self.imgproc.transform(image, self.transformations)
        labels = np.array(self._labels[idx]).astype(np.float32)
        return (transformed, labels)

    
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
