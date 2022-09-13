import os

import h5py
import hdf5storage

from scipy.io import loadmat, savemat, wavfile
from scipy.signal import stft, decimate, butter, sosfilt
from sklearn.model_selection import train_test_split
from resampy import resample
import librosa
import numpy as np
from tqdm.notebook import tqdm

import torch
import torchvision
from torchvision import transforms

import config


class DataHandler():

    """
    Data object for working with KRAKEN acoustic simulation output. 
    
    Attributes
    ----------
    p_t_noise: array-like, constains time-domain simulated calls from KRAKEN. Of shape (n_samples_per_example, n_examples).
    labels: array-like, labels for range, cb, cw, and zs, class, and SNR. Of shape (n_labels, n_examples).
    t_dec_min: array-like, time at which the dispersed acoustic signal ends. Of shape (n_examples,).
    t_dec_max: array-like, time at which the dispersed acoutic signal begins. Of shape (n_examples,).
    fs: float, sampling frequencey used in KRAKEN simulation.
    T: float, duration of KRAKEN simulation for an example.
    """

    def __init__(self, path, noise_only_path=None, sampled_noise_path=None, data_dir=None, mode='mat', fs_desired=None):

        """
        Initialize attributes relevant to KRAKEN simulation.

        Parameters
        ----------
        path: str, path to .mat file containing relevant data from the KRAKEN simulations.
        noise_only_path: str, path to .mat file containing noise only examples.
        data_dir: str, directory where all the data is stored.
        mode: str, either 'mat' or 'wav' and indicates the source of the time-domain data.
        fs_desired: int, desired fs for downsampling when loading data from wav files.
        """

        if mode != 'wav' and mode != 'mat':
            raise ValueError("mode must be set to either 'wav' or 'mat'.")

        if mode == 'wav' and data_dir is None:
            raise Exception("You must provide a data directory when in 'wav' mode.")

        self.mode = mode
        self.fs_desired = fs_desired
        self.__path = path
        self.__data_dir = data_dir
        self.__noise_only_path = noise_only_path
        self.__sampled_noise_path = sampled_noise_path
        self.__l2norm = False

        # Load data
        if mode == 'wav':
            self.__wav_2_mat()
        self.__load_mat()

    def __wav_2_mat(self):

        """
        Load data from wav files and saves it in a simmilar .mat format to the KRAKEN simulation output.
        This function assumes that the .wav files each contain 1 call and are named 'id-range|class' where 'range'
        is in kilometers. For example, '4-27_8|1.wav' for a call with id 4 that has label 27.8 km.
        """
        
        # gather names of all wav files in specified path
        files = os.listdir(self.__path)
        files = [f for f in files if ".wav" in f]
        num_files = len(files)

        # Load the calls
        calls = None
        for idx, f in enumerate(files):
            # Load an mean center a file
            fs, s = self.__load_wav(os.path.join(self.__path, f))
            s = s - s.mean()

            # Optionally downsample
            if self.fs_desired is not None and self.fs_desired != fs:
                s = resample(s, fs, self.fs_desired)
                fs = self.fs_desired

            if calls is None:
                calls = np.zeros((len(s), num_files))
                labels = np.zeros((2, num_files))
                T = len(s) / fs       

            calls[:,idx] = s

            r, c = f.split('|')
            c = int(c.split('.')[0])
            if c:
                labels[0, idx] = float(r.split('-')[1].replace('_','.'))*1000
            else:
                labels[0,idx] = -1
            labels[1, idx] = c

        # Save
        mdic = {u'p_t_r' : calls, u'labels' : labels, u'T' : T, u'fs' : fs}
        self.__path = os.path.join(self.__data_dir, 'wav_calls.mat')
        hdf5storage.write(mdic, '.', self.__path, matlab_compatible=True)

    def __load_mat(self):

        """
        Add results from KRAKEN simulation as class attributes.
        """
        # Load KRAKEN simulation data/.mat file from wav files
        mat_calls = h5py.File(self.__path, 'r')

        # Extract individual vectors
        self.p_t_noise = mat_calls['p_t_r'][:].T

        # Get theoretical start/end times if simulating
        if self.mode == 'mat':
            self.t_dec_min = np.squeeze(mat_calls['t_dec_min'][:])
            self.t_dec_max = np.squeeze(mat_calls['t_dec_max'][:])
        
        self.labels = np.squeeze(mat_calls['labels'][:].T).astype('float32')
        self.fs = float(np.squeeze(mat_calls['fs'][:]))
        self.T = float(np.squeeze(mat_calls['T'][:]))

        mat_calls.close()

        # # Load noise only data if possible
        if self.__noise_only_path is not None:
            mat_noise = h5py.File(self.__noise_only_path,'r')
            self.p_t_noise = np.concatenate((self.p_t_noise, mat_noise['p_t_noise_only'][:].T), axis=1)
            self.labels = np.concatenate((self.labels, np.squeeze(mat_noise['labels_noise_only'][:].T).astype('float64')), axis=1)
            mat_noise.close()

        # Number of examples and labels
        # gets the second dimension if simulated or just the length
        # of the range list if it's from wav
        try:
            self.n_examples = self.labels.shape[1]
        except:
            self.n_examples = len(self.labels)

    def __load_wav(self, path):

        """
        Load a wav file.
        Parameters
        ----------
        path: str, path to wav file.
        Return
        ------
        fs: int, sample rate.
        s: array (int), signal
        """

        fs, s = wavfile.read(path)

        return fs, s

    def __load_sampled_noise(self, n):
        
        # load noise
        mat_samp_noise = h5py.File(config.sample_noise, 'r')
        samp_noise = mat_samp_noise['data'][:]
        
        # get random indices
        inds = np.random.randint(0, samp_noise.shape[0], n)

        return samp_noise[inds,...]

    def save_sampled_noise(self):

        # Load KRAKEN simulation data/.mat file from wav files
        mat_samp_noise = h5py.File(self.__sampled_noise_path, 'r')
        samp_noise = mat_samp_noise['noise_from_data'][:].T
        fs = float(np.squeeze(mat_samp_noise['fs'][:]))

        # mean center
        samp_noise = samp_noise - samp_noise.mean(axis=0)

        # resample
        samp_noise = resample(samp_noise, fs, self.fs_desired, axis=0)

        # save
        with h5py.File(config.sample_noise, "w") as f:
            dset = f.create_dataset("data", data=samp_noise.T, chunks=(1,len(samp_noise[:,0])))

    def l2norm(self, x=None):

        """
        Mean-centers and L2 normalizes the time-domain signals.
        """
        if x is not None:
            x_mean = x.mean(axis=1)[:,np.newaxis]
            x = x - x_mean
            x = x / (np.sqrt(np.sum(x ** 2, axis=1))[:,np.newaxis])
            return x
        else:
            self.p_t_noise = self.p_t_noise - self.p_t_noise.mean(axis=0)
            self.p_t_noise = self.p_t_noise / np.sqrt(np.sum(self.p_t_noise ** 2, axis=0))

            self.__l2norm = True

    def create_signals(self, rand_shift=False, verbose=False):

        """
        Prepares time-domain signals for saving.

        Parameters
        ----------
        rand_shift: bool, randomly shifts the signal such that the call is not wrapped (if present) or just 
                    some random amount if the data is just noise.
        verbose: bool, print tqdm progress bar.

        Returns
        -------
        X: array-like, prepared data signals (n_examples, n_samples_per_example)
        y: array-like, data labels (n_examples, 5)
        """
        
        if self.mode == 'wav' and rand_shift == True:
            raise ValueError("Randome shift not supported for exerimental data.")
        for sig in tqdm(range(self.n_examples), disable=not verbose):

            # Random wrapping using start/end time of call or just length of data if it is noise
            if rand_shift and self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig], self.t_dec_min[sig], self.t_dec_max[sig])
            elif rand_shift and not self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig])

        # Transpose signals and labels
        self.y = self.labels.T.astype('float32')
        self.X = self.p_t_noise.T

        return self.X, self.y
             
    def create_spectrograms(self, fs=None, rand_shift=False, nperseg=31, noverlap=23, nfft=500, verbose=False):
        
        """
        Create spectrograms using dispersive calls from KRAKEN simulation. 

        Parameters
        ----------
        fs: float, sampling frequencey used in KRAKEN simulation.
        rand_shift: bool, whether or not to apply a random wrapping shift to the spectrogram.
        nperseg: int, number of samples per window.
        noverlap: int, number of samples overlap between windows.
        nfft: int, number of frequency 'bins' on the frequency axis of the spectrogram.
        verbose: bool, print example number as spectrograms are calculated.

        Returns
        -------
        X: array-like, calculated spectrograms of shape (n_examples, Zxx.shape[0], Zxx.shape[1]).
        y: array-like, labels of shape (n_labels, n_examples).
        f: array-like, frequency axis.
        t: array-like, time axis.
        """

        if fs is None:
            fs = self.fs

        if self.mode == 'wav' and rand_shift == True:
            raise ValueError("Randome shift not supported for exerimental data.")

        # Generate one spectrogram to get the dimensions for the current settings
        [_,_,Zxx] = stft(x=self.p_t_noise[:,0], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        self.X = np.zeros((self.n_examples, channels, Zxx.shape[0], Zxx.shape[1]), dtype='float32')

        for sig in tqdm(range(self.n_examples), disable=not verbose):

            # Random wrapping using start/end time of call or just length of data if it is noise
            if rand_shift and self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig], self.t_dec_min[sig], self.t_dec_max[sig])
            elif rand_shift and not self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig])

                [f, t, Zxx] = stft(x=self.p_t_noise[:,sig], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                log_spec = np.flipud(10*np.log10(np.abs(Zxx)**2))
                if size is not None:
                    log_spec = resize(log_spec, size, anti_aliasing=True)
                
                # Save data
                log_spec = log_spec[np.newaxis, :,:]
                self.X[sig,:,:,:] = log_spec

        # Make first dimension the number of examples
        self.y = self.labels.T.astype('float32')

        return self.X, self.y, f, t

    def __wrap_signal_random(self, x, t_min=None, t_max=None):
        
        """
        Randomly wraps a signal to simulate a call appearing in different parts of the window
        while protecting against landing the call in between the end and beginning of the iamge.
        Parameters
        ----------
        x: array-like, spectrograms to wrap
        t_min: float, end time of call
        t_max: float, start time of call
        Returns
        -------
        shifted_spec: array-like, wrapped version of original spectrogram
        shifted_spec: 
        """

        # Calculate minimum and maximum shifts without catching
        # the call between the end and the beinning of the window
        min_shift = 0

        if t_min is None or t_max is None:
            max_shift = len(x)
        else:
            max_shift = np.floor((self.T - (t_min - t_max)))*self.fs

        # Calculate the random shift amount and shift
        if max_shift == min_shift:
            rand_shift = 0
        else:
            rand_shift = np.random.randint(min_shift, max_shift)
        shifted_sig = np.roll(x, rand_shift, 0)

        return shifted_sig

    def save_h5(self, path):

        """
        Save spectrograms and labels in h5 format.
        Parameters
        ----------
        path: str, path where the h5 file will be saved.
        """

        if self.X is None or self.y is None:
            raise Exception("Spectram data has not yet been created.")

        with h5py.File(path, 'w') as f:
            f.create_dataset("data", data=self.X)
            f.create_dataset("labels", data=self.y)

    def save_split_h5(self, train_path, val_path, test_path=None, val_size=0.2, test_size=0, train_size=0.8, random_state=None, shuffle=True, stratify=None):
        
        """
        Save data and labels in h5 format and split into train, validation, and test set.

        Parameters
        ----------
        train_path: str, path where the train h5 file will be saved.
        val_path: str, path where the validation h5 file will be saved.
        test_path: str, path where the test h5 file will be saved.
        val_size: float, decimal percentage of data for validation.
        test_size: float, decimal percentage of data for testing.
        train_size: float, decimal percentage of data for training.
        random_state: int, RandomState instance or None, Controls the 
                      shuffling applied to the data before applying the 
                      split. Pass an int for reproducible output across 
                      multiple function calls.
        shuffle: bool, Whether or not to shuffle the data before splitting. 
                 If shuffle=False then stratify must be None.
        stratify: array-like, If not None, data is split in a stratified 
                  fashion, using this as the class labels.
        """

        if self.X is None or self.y is None:
            raise Exception("Spectram data has not yet been created.")

        if len(self.X.shape) == 3:
            maxshape = (None, self.X.shape[0], self.X.shape[1], self.X.shape[2])
            chunks = (1, self.X.shape[0], self.X.shape[1], self.X.shape[2])
        else:
            maxshape = (None, self.X.shape[1])
            chunks = (1, self.X.shape[1])

        print("Splitting Data...")
        # Train/val split
        inds = np.arange(self.X.shape[0])
        X_train, X_val, y_train, y_val = train_test_split(inds, inds, test_size=val_size, train_size=train_size+test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        
        # Test split from training data
        if test_size != 0:
            test_size = test_size / (train_size + test_size)
            train_size = train_size / (train_size + test_size)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

        # Chunk size to use when saving data to .h5 file in batches
        c = 2000

        print("Saving Training Data...")
        remain = (self.X[X_train].shape[0] % c)
        top = self.X[X_train].shape[0] - remain
        with h5py.File(train_path, 'a') as f:
            i = 0
            f.create_dataset("data", data=self.X[X_train][:i+c,...], chunks=chunks, maxshape=maxshape)
            f.create_dataset("labels", data=self.y[y_train][:i+c,...], chunks=(1,6), maxshape=(None,6))
            i += c
            while True:
                if i == top:
                    f["data"].resize((f["data"].shape[0] + remain), axis=0)
                    f["data"][i:,...] = self.X[X_train][i:,...]

                    f["labels"].resize((f["labels"].shape[0] + remain), axis=0)
                    f["labels"][i:,...] = self.y[y_train][i:,...]
                    break
                else:
                    f["data"].resize((f["data"].shape[0] + c), axis=0)
                    f["data"][i:i+c,...] = self.X[X_train][i:i+c,...]

                    f["labels"].resize((f["labels"].shape[0] + c), axis=0)
                    f["labels"][i:i+c,...] = self.y[y_train][i:i+c,...]
                i += c
                print(f["data"].shape)

        if test_size is not None:
            print("Saving Test Data...")
            test_noise = self.__load_sampled_noise(len(X_test))
            if self.__l2norm:
                test_noise = self.l2norm(test_noise)
            test_noise_labels = -1*np.ones((test_noise.shape[0], 6))
            test_noise_labels[:,4] = 0
            X_test_noise = np.concatenate((self.X[X_test], test_noise), axis=0)
            y_test_noise = np.concatenate((self.y[y_test], test_noise_labels), axis=0)
            remain = (X_test_noise.shape[0] % c)
            top = X_test_noise.shape[0] - remain
            with h5py.File(test_path, 'a') as f:
                i = 0
                f.create_dataset("data", data=X_test_noise[:i+c,...], chunks=chunks, maxshape=maxshape)
                f.create_dataset("labels", data=y_test_noise[:i+c,...], chunks=(1,6), maxshape=(None,6))
                i += c
                while True:
                    if i == top:
                        f["data"].resize((f["data"].shape[0] + remain), axis=0)
                        f["data"][i:,...] = X_test_noise[i:,...]

                        f["labels"].resize((f["labels"].shape[0] + remain), axis=0)
                        f["labels"][i:,...] = y_test_noise[i:,...]
                        break
                    else:
                        f["data"].resize((f["data"].shape[0] + c), axis=0)
                        f["data"][i:i+c,...] = X_test_noise[i:i+c,...]

                        f["labels"].resize((f["labels"].shape[0] + c), axis=0)
                        f["labels"][i:i+c,...] = y_test_noise[i:i+c,...]
                    i += c
                    print(f["data"].shape)

        print("Saving Validation Data")
        remain = (self.X[X_val].shape[0] % c)
        top = self.X[X_val].shape[0] - remain
        val_noise = self.__load_sampled_noise(len(X_val))
        if self.__l2norm:
            val_noise = self.l2norm(val_noise)
        val_noise_labels = -1*np.ones((val_noise.shape[0], 6))
        val_noise_labels[:,4] = 0
        X_val_noise = np.concatenate((self.X[X_val], val_noise), axis=0)
        y_val_noise = np.concatenate((self.y[y_val], val_noise_labels), axis=0)
        remain = (y_test_noise.shape[0] % c)
        top = y_test_noise.shape[0] - remain
        with h5py.File(val_path, 'a') as f:
            i = 0
            f.create_dataset("data", data=X_val_noise[:i+c,...], chunks=chunks, maxshape=maxshape)
            f.create_dataset("labels", data=y_val_noise[:i+c,...], chunks=(1,6), maxshape=(None,6))
            i += c
            while True:
                if i == top:
                    f["data"].resize((f["data"].shape[0] + remain), axis=0)
                    f["data"][i:,...] = X_val_noise[i:,...]

                    f["labels"].resize((f["labels"].shape[0] + remain), axis=0)
                    f["labels"][i:,...] = y_val_noise[i:,...]
                    break
                else:
                    f["data"].resize((f["data"].shape[0] + c), axis=0)
                    f["data"][i:i+c,...] = X_val_noise[i:i+c,...]

                    f["labels"].resize((f["labels"].shape[0] + c), axis=0)
                    f["labels"][i:i+c,...] = y_val_noise[i:i+c,...]
                i += c
                print(f["data"].shape)
