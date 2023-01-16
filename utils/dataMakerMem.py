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

    def __init__(self, path, noise_only_path=None, sampled_noise_path=None, data_dir=None, mode='mat', fs_desired=None, l2norm=True):

            """
            Initialize attributes relevant to KRAKEN simulation.

            Parameters
            ----------
            path: str, path to .mat file containing relevant data from the KRAKEN simulations.
            noise_only_path: str, path to .mat file containing noise only examples.
            sampled_noise_path: str, path to .mat file containing experimental noise examples. Must
                                be the same duration as the simulated signals, but can have a higher
                                sampling frequency.
            data_dir: str, directory where all the data is stored.
            mode: str, either 'mat' or 'wav' and indicates the source of the time-domain data.
            fs_desired: int, desired fs for downsampling when loading data from wav files.
            l2norm: bool, whether or not to l2 norm each of the time-domain signals.
            """

            if mode != 'wav' and mode != 'mat':
                raise ValueError("mode must be set to either 'wav' or 'mat'.")

            if mode == 'wav' and data_dir is None:
                raise Exception("You must provide a data directory when in 'wav' mode.")

            self.mode = mode
            self.fs_desired = fs_desired
            self.l2norm = l2norm
            self.__path = path
            self.__data_dir = data_dir
            self.__noise_only_path = noise_only_path
            self.__sampled_noise_path = sampled_noise_path

            # Load data
            if mode == 'wav':
                self.__wav_2_mat()

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

    def __load_mat(self, start_ind=0, end_ind=-1):

        """
        Add results from KRAKEN simulation as class attributes.

        Parameters
        ----------
        start_ind: int, start index for stored examples
        end_ind: int, end index for stored examples
        """
        # Load KRAKEN simulation data/.mat file from wav files
        mat_calls = h5py.File(self.__path, 'r')

        # Extract individual vectors
        self.p_t_noise = mat_calls['p_t_r'][start_ind:end_ind,:].T

        # Get theoretical start/end times if simulating
        if self.mode == 'mat':
            self.t_dec_min = np.squeeze(mat_calls['t_dec_min'][start_ind:end_ind])
            self.t_dec_max = np.squeeze(mat_calls['t_dec_max'][start_ind:end_ind])
        
        self.labels = np.squeeze(mat_calls['labels'][start_ind:end_ind,:].T).astype('float32')
        self.fs = float(np.squeeze(mat_calls['fs'][:]))
        self.T = float(np.squeeze(mat_calls['T'][:]))

        mat_calls.close()

        # # Load noise only data if possible
        if self.__noise_only_path is not None:
            mat_noise = h5py.File(self.__noise_only_path,'r')
            self.p_t_noise = np.concatenate((self.p_t_noise, mat_noise['p_t_noise_only'][start_ind:end_ind,:].T), axis=1)
            self.labels = np.concatenate((self.labels, np.squeeze(mat_noise['labels_noise_only'][start_ind:end_ind,:].T).astype('float64')), axis=1)
            mat_noise.close()

        # Number of examples and labels
        # gets the second dimension if simulated or just the length
        # of the range list if it's from wav
        # try:
        #     self.n_examples = self.labels.shape[1]
        # except:
        #     self.n_examples = len(self.labels)

    def __load_wav(self, path):

        """
        Load a wav file.

        Parameters
        ----------
        path: str, path to wav file.
        
        Returns
        ------
        fs: int, sample rate.
        s: array (int), signal
        """

        fs, s = wavfile.read(path)

        return fs, s

    def __load_sampled_noise(self, n):

        """
        Load a collection of sampled noise examples.

        Parameters
        ----------
        n: int, number of examples to load.

        Returns
        -------
        samp_noise[inds,...]: array-like, noise examples (n, samples/call)
        """
        
        # load noise
        mat_samp_noise = h5py.File(config.sample_noise, 'r')
        samp_noise = mat_samp_noise['data'][:]
        
        # get random indices
        inds = np.random.randint(0, samp_noise.shape[0], n)

        return samp_noise[inds,...]

    def save_sampled_noise(self):

        """
        Load saved noise examples, mean-center, and save as .h5
        """

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

    def __l2norm(self, x, axis=0):

        """
        Mean-centers and L2 normalizes the time-domain signals.

        Parameters
        ----------
        x: array-like, (samples/call, num calls)

        Returns
        -------
        array-like, mean-centered calls (samples/call, num_calls)
        """

        x = x - x.mean(axis=axis, keepdims=True)
        return x / np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))

    def __get_dataset_size(self):
        
        """
        Get total size of the dataset.

        Returns
        -------
        int, size of dataset
        """

        # Load KRAKEN simulation data/.mat file from wav files
        mat_calls = h5py.File(self.__path, 'r')

        return mat_calls['t_dec_max'].size, mat_calls['labels'].size // mat_calls['t_dec_max'].size

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

    def create_and_save_signals(self, split, train_size=0.8, test_size=0.1, val_size=0.1, save_chunk_size=10000, rand_shift=False, verbose=False):

        """
        Prepares time-domain signals for saving.

        Parameters
        ----------
        rand_shift: bool, randomly shifts the signal such that the call is not wrapped (if present) or just 
                    some random amount if the data is just noise.
        split: bool, whether or not to do train/test split or save all data to one file.
        save_chunk_size: int, number of signals to preprocess and save at once.
        verbose: bool, print tqdm progress bar.

        Returns
        -------
        X: array-like, prepared data signals (n_examples, n_samples_per_example)
        y: array-like, data labels (n_examples, 5)
        """

        if self.mode == 'wav' and rand_shift == True:
            raise ValueError("Randome shift not supported for exerimental data.")

        self.n_examples, num_labels = self.__get_dataset_size()

        bar = tqdm(total=self.n_examples, disable=not verbose)
        for c in range(0, self.n_examples, save_chunk_size):
            
            # get size of next chunk to save
            if self.n_examples - c < save_chunk_size:
                chunk_size = self.n_examples - c
                start_ind = c
                end_ind = c + chunk_size
            else:
                chunk_size = save_chunk_size
                start_ind = c
                end_ind = c + chunk_size

            # get chunk of data
            self.__load_mat(start_ind, end_ind)

            # random shift if requested
            for sig in range(chunk_size):
                # Random wrapping using start/end time of call or just length of data if it is noise
                if rand_shift and self.labels[4,sig]:
                    self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig], self.t_dec_min[sig], self.t_dec_max[sig])
                elif rand_shift and not self.labels[4,sig]:
                    self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig])

                # iterate tqdm
                bar.update(1)

            # l2norm if requsted
            if self.l2norm:
                self.p_t_noise = self.__l2norm(self.p_t_noise)

            # make first index of data and labels the example index
            self.p_t_noise = self.p_t_noise.T
            self.labels = self.labels.T.astype('float32')
            
            # split into train, val, and optionally test sets
            inds = np.arange(self.p_t_noise.shape[0])
            X_train, X_val, y_train, y_val = train_test_split(inds, inds, test_size=val_size, train_size=train_size+test_size, shuffle=True)
            if test_size > 0:
                test_size_new = round(test_size / (1.0 - val_size), 5)
                train_size_new = round(1.0 - test_size_new, 5)
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size_new, train_size=train_size_new, shuffle=True)

            # get maxshapes and chink sizes for data and labels
            if c == 0:
                sig_maxshape = (None, self.p_t_noise.shape[1])
                lab_maxshape = (None, num_labels)
                sig_chunks = (1, self.p_t_noise.shape[1])
                lab_chunks = (1, num_labels)

            # save train split
            with h5py.File(config.train_data, 'a') as f:

                if c == 0:
                    f.create_dataset("data", data=self.p_t_noise[X_train], chunks=sig_chunks, maxshape=sig_maxshape)
                    f.create_dataset("labels", data=self.labels[y_train], chunks=lab_chunks, maxshape=lab_maxshape)
                    train_count = len(X_train)
                else:
                    f["data"].resize((f["data"].shape[0] + len(X_train)), axis=0)
                    f["data"][train_count:,...] = self.p_t_noise[X_train]
                    
                    f["labels"].resize((f["labels"].shape[0] + len(X_train)), axis=0)
                    f["labels"][train_count:,...] = self.labels[X_train]
                    train_count += len(X_train)
            
            # optionally save test split
            if test_size > 0:
                with h5py.File(config.test_data, 'a') as f:
                    if c == 0:
                        noise_test = self.__load_sampled_noise(len(X_test))
                        if self.l2norm:
                            noise_test = self.__l2norm(noise_test, axis=1)
                        if rand_shift:
                            for sig in range(len(noise_test)):
                                noise_test[sig,:] = self.__wrap_signal_random(noise_test[sig,:])

                        f.create_dataset("data", data=np.concatenate((self.p_t_noise[X_test], noise_test), axis=0), chunks=sig_chunks, maxshape=sig_maxshape)
                        noise_labels_test = -1*np.ones(self.labels[y_test].shape)
                        noise_labels_test[:,4] = 0 
                        f.create_dataset("labels", data=np.concatenate((self.labels[y_test], noise_labels_test), axis=0), chunks=lab_chunks, maxshape=lab_maxshape)
                        test_count = 2*len(X_test)
                    else:
                        noise_test = self.__load_sampled_noise(len(X_test))
                        if self.l2norm:
                            noise_test = self.__l2norm(noise_test, axis=1)
                        if rand_shift:
                            for sig in range(len(noise_test)):
                                noise_test[sig,:] = self.__wrap_signal_random(noise_test[sig,:])

                        f["data"].resize((f["data"].shape[0] + 2*len(X_test)), axis=0)
                        f["data"][test_count:,...] = np.concatenate((self.p_t_noise[X_test], noise_test), axis=0)
                    
                        f["labels"].resize((f["labels"].shape[0] + 2*len(X_test)), axis=0)
                        noise_labels_test = -1*np.ones(self.labels[y_test].shape)
                        noise_labels_test[:,4] = 0 
                        f["labels"][test_count:,...] = np.concatenate((self.labels[y_test], noise_labels_test), axis=0)
                        test_count += 2*len(X_test)

            # save validation split
            with h5py.File(config.val_data, 'a') as f:
                if c == 0:
                    noise_val = self.__load_sampled_noise(len(X_test))
                    if self.l2norm:
                        noise_val = self.__l2norm(noise_val, axis=1)
                    if rand_shift:
                        for sig in range(len(noise_val)):
                            noise_val[sig,:] = self.__wrap_signal_random(noise_val[sig,:])

                    f.create_dataset("data", data=np.concatenate((self.p_t_noise[X_val], noise_val),axis=0), chunks=sig_chunks, maxshape=sig_maxshape)
                    noise_labels_val = -1*np.ones(self.labels[y_val].shape)
                    noise_labels_val[:,4] = 0 
                    f.create_dataset("labels", data=np.concatenate((self.labels[y_val], noise_labels_val), axis=0), chunks=lab_chunks, maxshape=lab_maxshape)
                    val_count = 2*len(X_val)
                else:
                    noise_val = self.__load_sampled_noise(len(X_val))
                    if self.l2norm:
                        noise_val = self.__l2norm(noise_val, axis=1)
                    if rand_shift:
                        for sig in range(len(noise_val)):
                            noise_val[sig,:] = self.__wrap_signal_random(noise_val[sig,:])

                    f["data"].resize((f["data"].shape[0] + 2*len(X_val)), axis=0)
                    f["data"][val_count:,...] = np.concatenate((self.p_t_noise[X_val], noise_val), axis=0)
                    
                    f["labels"].resize((f["labels"].shape[0] + 2*len(X_val)), axis=0)
                    noise_labels_val = -1*np.ones(self.labels[y_val].shape)
                    noise_labels_val[:,4] = 0 
                    f["labels"][val_count:,...] = np.concatenate((self.labels[y_val], noise_labels_val), axis=0)
                    val_count += 2*len(X_val)        
        
        # close tqdm bar
        bar.close()
