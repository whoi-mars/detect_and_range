import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import h5py
import numpy as np
import os
from data_augmentation import FrequencyBandZeroing
import config

class Gunshot(data.Dataset):
    """Dataset to load data from the Oxford pet dataset .h5 files
    :param dir_path: path to directory containing data (e.g. train, test or val)
    :type dir_path: str
    """

    def __init__(self, dir_path, max_range, transform=None, squeeze=False):

        super(Gunshot, self).__init__()

        self.dir_path = dir_path
        self.transform = transform
        self.squeeze = squeeze
        self.inputs, self.imsize = self._load_h5_file_with_data()
        self.max_range = max_range

    def __getitem__(self, index):

        inputs = self._from_numpy(self.inputs['data'][index])
        class_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,4]]))
        range_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,0]]))
        range_targets[range_targets != -1] = range_targets / self.max_range

        if self.transform is not None:
            inputs = self.transform(inputs)
        
        if self.squeeze:
            inputs = inputs.squeeze()

        return inputs, range_targets, class_targets

    def __len__(self):
        return self.inputs['data'].shape[0]

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5_file_with_data(self):
        file = h5py.File(self.dir_path)
        return dict(data=file['data'][:], labels=file['labels'][:]), file['data'].shape[2:]

class Warped(data.Dataset):
    """Dataset to load data from the Oxford pet dataset .h5 files
    :param dir_path: path to directory containing data (e.g. train, test or val)
    :type dir_path: str
    """

    def __init__(self, dir_path, max_range, transform=None, squeeze=False):

        super(Warped, self).__init__()

        self.dir_path = dir_path
        self.transform = transform
        self.squeeze = squeeze
        self.inputs, self.imsize = self._load_h5_file_with_data()
        self.max_range = max_range

    def __getitem__(self, index):

        inputs = self._from_numpy(self.inputs['data'][index])
        class_targets = 1.
        range_targets = self._from_numpy(np.asarray(self.inputs['labels'][index]))
        range_targets = range_targets / self.max_range

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.squeeze:
            inputs = inputs.squeeze()

        return inputs, range_targets, class_targets

    def __len__(self):
        return self.inputs['data'].shape[0]

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5_file_with_data(self):
        file = h5py.File(self.dir_path)
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

def get_image_transforms():
    
    """
    Make dictionary of transforms for sets used for training/evaluation
    
    Returns
    -------
    transforms: dict, transforms for training/evaluation datasets
    """
    
    transform_eval = transforms.Compose([
        transforms.Normalize([config.mu], [config.std]),
    ])

    transform_train = transforms.Compose([
        transforms.Normalize([config.mu], [config.std]),
        FrequencyBandZeroing(max_freq_width=30)
    ])

    transform_dict = {'train': transform_train, 'eval': transform_eval}

    return transform_dict

def get_dataloaders(data_dir, batch_size, max_range, shuffle=True, transform=None, squeeze=False):
    
    """
    Make dictionary of dataloaders for the train, validation, and test sets.
    
    Parameters
    ----------
    data_dir: str, directory that contains the data .h5 files
    batch_size: int, numbere of examples per batch
    shuffle: bool, whether or not to shuffle the training set
    transform: dict, dictionary of transforms with keys 'train' and 'eval'
    
    Returns
    -------
    dataloaders: dict, dictionary with data loaders for each dataset
    """
    
    # Get the similar parts of the train/validation/test file names
    name_base_train = "_".join((config.train_data.split('/')[-1].split('_')[:-1])) + "_"
    name_base_val = "_".join((config.val_data.split('/')[-1].split('_')[:-1])) + "_"
    name_base_test = "_".join((config.test_data.split('/')[-1].split('_')[:-1])) + "_"
    
    base_list = [name_base_train, name_base_val, name_base_test]
    
    if all(base == base_list[0] for base in base_list):
        # Transform dictionary
        data_transforms = {
            'train': transform['train'] if transform is not None else transform,
            'val': transform['eval'] if transform is not None else transform,
            'test': transform['eval'] if transform is not None else transform
        }

        # Create dataset
        datasets = {x: Gunshot(dir_path=os.path.join(config.data_dir, 'grid_data_atten_big_{}.h5'.format(x)), max_range=config.max_range, transform=data_transforms[x], squeeze=squeeze) for x in data_transforms.keys()}

        # Make dataloaders
        dataloaders = {x: data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle) for x in data_transforms.keys()}

        return dataloaders
    else:
        raise ValueError("According to their names, the data sets you are using to train appear to be inconsistent")

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