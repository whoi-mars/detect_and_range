import os

import torch
import torchvision
from torchvision import transforms
from torch.utils import data

from scipy.signal import stft, butter, sosfilt
import numpy as np
import h5py
import warnings

from utils.data_augmentation import FrequencyBandZeroing, Normalize1DChannel, ZeroOneNorm
import config

class Gunshot(data.Dataset):

    """
    Dataset to load KRAKEN simulated data. It expects the data to be stored in a .h5 file format
    with 'data' and 'labels' keys. The 'data' should be of shape (# examples, # samples_per_example)
    while the labels are of shape (# examples, 6) where each row contains (range [m], cb [m/s], cw [m/s],
    zs [m], class label [either 0 or 1] and SNR [dB]).

    Parameters
    ----------
    dir_path: str, path to the .h5 file containing data.
    max_range: float, maximum range value (in meters) included in the simulated data set. This value will
               be used to normalize the labels.
    transform: torchvision.transforms.Compose, transformation composition to apply.
    squeeze: bool, whether or not to squeeze the unit channel dimension.
    from_time: bool, whether or not the data is a timeseries or needs to be transformed into spectrograms.
    hpf: bool, apply high-pass filter while loading data.

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
        
        # Preprocess data
        inputs = self.inputs['data'][[index]]
        if self.hpf:
            inputs = self._hpf(inputs)
        if self.from_time:
            inputs = to_spect(inputs).squeeze(axis=1).copy()

        inputs = self._from_numpy(inputs)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.squeeze:
            inputs = inputs.squeeze()

        # Preprocess labels
        class_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,4]]))
        range_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,0]]))
        #snr_vals = self._from_numpy(np.asarray([self.inputs['labels'][index,5]]))
        range_targets[range_targets != -1] = range_targets / self.max_range

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
    hpf: bool, apply high-pass filter while loading data.

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

        # Preprocess inputs
        inputs = self.inputs['data'][[index]]
        if self.hpf:
            inputs = self._hpf(inputs)
        if self.from_time:
            inputs = to_spect(inputs).squeeze(axis=1).copy()
        
        inputs = self._from_numpy(inputs)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.squeeze:
            inputs = inputs.squeeze()

        # Preprocess labels        
        class_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,1]]))
        range_targets = self._from_numpy(np.asarray([self.inputs['labels'][index,0]]))
        range_targets[range_targets != -1] = range_targets / self.max_range

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
    max_range: float, maximum range of call to be used in normalization
    shuffle: bool, whether or not to shuffle the training set
    transform: dict, dictionary of transforms with keys 'train' and 'eval'
    squeeze: bool, whether or not to squeeze the unit channel dimension.
    hpf: bool, apply high-pass filter while loading data.
    
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
        
        # Get base name
        name_base = name_base_train
        
        # Transform dictionary
        data_transforms = {
            'train': transform['train'] if transform is not None else transform,
            'val': transform['eval'] if transform is not None else transform,
            'test': transform['eval'] if transform is not None else transform
        }

        # Create dataset
        datasets = {x: Gunshot(dir_path=os.path.join(config.data_dir, name_base+'{}.h5'.format(x)), max_range=config.max_range, transform=data_transforms[x], squeeze=squeeze, hpf=hpf) for x in data_transforms.keys()}

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