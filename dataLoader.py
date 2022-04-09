import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data

def load_h5(path):

    """
    Loads spectrogram data and labels saved in an h5 format.
    Parameters
    ----------
    path: str, path where the h5 file is saved.
    Returns
    ----------
    X: array-like, calculated spectrograms of shape (n_examples, Zxx.shape[0], Zxx.shape[1]).
    y: array-like, labels of shape (n_labels, n_examples).
    """

    with h5py.File(path, 'r') as f:
        X = f["data"][:]
        y = f["labels"][:]

    return X, y

class Gunshot(data.Dataset):
    """Dataset to load data from the Oxford pet dataset .h5 files
    :param dir_path: path to directory containing data (e.g. train, test or val)
    :type dir_path: str
    """

    def __init__(self, dir_path, transform=None):

        super(Gunshot, self).__init__()

        self.dir_path = dir_path
        self.transform = transform
        self.inputs, self.imsize = self._load_h5_file_with_data()
        # self.targets = {
        #     task: self._load_h5_file_with_data(self.task_to_file[task]) for task in tasks if task in self.task_to_file
        # }

        #self.transform = Compose([self._from_numpy, self._permute_tf_to_torch]+transform)
        #self.target_transforms = self._prepare_default_target_transforms(target_transforms)

    def __getitem__(self, index):

        inputs = self._from_numpy(self.inputs['data'][index])
        targets = self._from_numpy(self.inputs['labels'][index])

        if self.transform is not None:
            inputs = self.transform(inputs)

        return inputs, targets

    def __len__(self):
        return self.inputs['data'].shape[0]

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5_file_with_data(self):
        file = h5py.File(self.dir_path)
        # key = list(file.keys())[0]
        # data = file[key]
        # return dict(file=file, data=data)
        return dict(data=file['data'], labels=file['labels']), file['data'].shape[2:]


class RandomBatchSampler(data.Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)

def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return data.DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=data.BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )