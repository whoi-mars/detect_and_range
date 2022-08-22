import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from scipy.signal import stft, butter, sosfilt
import h5py
import numpy as np
import os
from data_augmentation import FrequencyBandZeroing, Normalize1DChannel, ZeroOneNorm
import config
import warnings

def to_spect(x):

    """
    Calculates a batch of spectrograms from time-domain signals.

    Parameters
    ----------
    x: array-like, matrix of time-domain signals from which to calculate spectrograms. Should be of
       shape (# signals, # samples / signal)
    
    Returns
    -------
    log_spec: array-like, resulting spectrograms. Of shape (# signals, 1, # frequency bins, # time bins)
    """

    # Calculate STFTs
    [f, t, Zxx] = stft(x=x, fs=config.fs, nperseg=config.nperseg, noverlap=config.noverlap, nfft=config.nfft)
    # Get dB power of each spectrogram and put matrix in correct orientation
    log_spec = np.flip(10*np.log10(np.abs(Zxx)**2), axis=1)
    # Add channel dimension
    log_spec = log_spec[:, np.newaxis, ...]
    return log_spec

class Gunshot(data.Dataset):

    """
    Dataset to load KRAKEN simulated data. It expects the data to be stored in a .h5 file format
    with 'data' and 'labels' keys. The 'data' should be of shape (# examples, # samples_per_example)
    while the labels are of shape (# examples, 5) where each row contains (range [m], cb [m/s], cw [m/s],
    zs [m], and class label [either 0 or 1]).

    Parameters
    ----------
    dir_path: str, path to the .h5 file containing data.
    max_range: float, maximum range value (in meters) included in the simulated data set. This value will
               be used to normalize the labels.
    transform: torchvision.transforms.Compose, transformation composition to apply.
    squeeze: bool, whether or not to squeeze the unit channel dimension.
    from_time: bool, whether or not the data is a timeseries or needs to be transformed into spectrograms.

    Returns
    -------
    inputs: array-like, spectrograms.
    range_targets: float, ranges normalized to [0, 1]
    class_targets: int, class labels
    """

    def __init__(self, dir_path, max_range, transform=None, squeeze=False, from_time=True, hpf=False):

        super(Gunshot, self).__init__()

        self.dir_path = dir_path
        self.transform = transform
        self.squeeze = squeeze
        self.inputs, self.imsize = self._load_h5_file_with_data()
        if from_time:
            self.imsize = to_spect([self.inputs['data'][0]]).shape[2:]
        self.max_range = max_range
        self.from_time = from_time
        self.hpf = hpf
        if hpf:
            self.sos = butter(config.order, config.fc, 'highpass', fs=config.fs, output='sos')

    def __getitem__(self, index):
        
        inputs = self.inputs['data'][[index]]
        if self.hpf:
            inputs = self._hpf(inputs)
        if self.from_time:
            inputs = to_spect(inputs).squeeze(axis=1).copy()
        inputs = self._from_numpy(inputs)
        class_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,4]]))
        range_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,0]]))
        #snr_vals = self._from_numpy(np.asarray([self.inputs['labels'][index,5]]))
        range_targets[range_targets != -1] = range_targets / self.max_range

        if self.transform is not None:
            inputs = self.transform(inputs)
        
        if self.squeeze:
            inputs = inputs.squeeze()

        return inputs, range_targets, class_targets #, snr_vals

    def __len__(self):
        return self.inputs['data'].shape[0]

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

    def _load_h5_file_with_data(self):
        file = h5py.File(self.dir_path)
        return dict(data=file['data'][:], labels=file['labels'][:]), file['data'].shape[2:]

    def _hpf(self, tensor):
        return sosfilt(self.sos, tensor)    

class Warped(data.Dataset):

    """
    Dataset to load examples from experimental data which have been ranged using the warping technique.
    This dataset object is the exact same as the Gunshot one above except for the fact that it expects
    the 'labels' in the .h5 file to be a 1D array of ranges (in meters), and the class label is generated
    within the dataset object and is always 1.

    Parameters
    ----------
    dir_path: str, path to the .h5 file containing data.
    max_range: float, maximum range value (in meters) included in the simulated data set. This value will
               be used to normalize the labels.
    transform: torchvision.transforms.Compose, transformation composition to apply.
    squeeze: bool, whether or not to squeeze the unit channel dimension.
    from_time: bool, whether or not the data is a timeseries or needs to be transformed into spectrograms.

    Returns
    -------
    inputs: array-like, spectrograms.
    range_targets: float, ranges normalized to [0, 1]
    class_targets: int, class labels
    """

    def __init__(self, dir_path, max_range, transform=None, squeeze=False, from_time=True, hpf=False):

        super(Warped, self).__init__()

        self.dir_path = dir_path
        self.transform = transform
        self.squeeze = squeeze
        self.inputs, self.imsize = self._load_h5_file_with_data()
        if from_time:
            self.imsize = to_spect([self.inputs['data'][0]]).shape[2:]
        self.max_range = max_range
        self.from_time = from_time
        self.hpf = hpf
        if hpf:
            self.sos = butter(config.order, config.fc, 'highpass', fs=config.fs, output='sos')

    def __getitem__(self, index):

        inputs = self.inputs['data'][[index]]
        if self.hpf:
            inputs = self._hpf(inputs)
        if self.from_time:
            inputs = to_spect(inputs).squeeze(axis=1).copy()
        inputs = self._from_numpy(inputs)
        class_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,1]]))
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
        return dict(data=file['data'], labels=file['labels']), file['data'].shape[2:]
    
    def _hpf(self, tensor):
        return sosfilt(self.sos, tensor)

def get_image_transforms():
    
    """
    Make dictionary of transforms for sets used for training/evaluation
    
    Returns
    -------
    transforms: dict, transforms for training/evaluation datasets
    """
    
    transform_eval = transforms.Compose([
        Normalize1DChannel(config.mu_list_l2, config.std_list_l2),
    ])

    transform_train = transforms.Compose([
        Normalize1DChannel(config.mu_list_l2, config.std_list_l2),
        FrequencyBandZeroing(max_freq_width=30, max_t_width=30, num_f=1, num_t=0)
    ])

    transform_dict = {'train': transform_train, 'eval': transform_eval}

    return transform_dict

def get_dataloaders(data_dir, batch_size, max_range, shuffle=True, transform=None, squeeze=False, hpf=False):
    
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
        datasets = {x: Gunshot(dir_path=os.path.join(config.data_dir, 'grid_data_atten_big_{}.h5'.format(x)), max_range=config.max_range, transform=data_transforms[x], squeeze=squeeze, hpf=hpf) for x in data_transforms.keys()}

        # Make dataloaders
        dataloaders = {x: data.DataLoader(datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=32) for x in data_transforms.keys()}

        return dataloaders
    else:
        warnings.warn("According to their names, the data sets you are using to train appear to be inconsistent")

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

class RandomBatchSampler(data.Sampler):
    """
    Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset, dataset you want to batch
    batch_size: int, batch size
    
    Returns
    -------
    generator object of shuffled batch indices
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
    """
    Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset, a dataset that loads data from .h5 files
    batch_size: int, size of data to batch
    drop_last: bool, flag to indicate if last batch will be dropped (if size < batch_size)
    
    Returns
    -------
    torch.utils.data.DataLoader, dataloading that queries from data using shuffled batches
    """
    return data.DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=data.BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )