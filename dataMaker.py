from scipy.io import loadmat
from scipy.signal import stft
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.notebook import tqdm
from skimage.transform import resize
import h5py
import os

class DataHandler():

    """
    Data object for working with KRAKEN acoustic simulation output. 
    
    Attributes
    ----------
    p_t_noise: array-like, constains time-domain simulated calls from KRAKEN. Of shape (n_samples_per_example, n_examples).
    labels: array-like, labels for range, cb, cw, and zs. Of shape (n_labels, n_examples).
    t_dec_min: array-like, time at which the dispersed acoustic signal ends. Of shape (n_examples,).
    t_dec_max: array-like, time at which the dispersed acoutic signal begins. Of shape (n_examples,).
    fs: float, sampling frequencey used in KRAKEN simulation.
    T: float, duration of KRAKEN simulation for an example.
    """

    def __init__(self, path, noise_only_path=None):

        """
        Initialize attributes relevant to KRAKEN simulation.
        Parameters
        ----------
        path: str, path to .mat file containing relevant data from the KRAKEN simulations
        """
        
        self.__path = path
        self.__noise_only_path = noise_only_path

        # Load KRAKEN data
        self.__load_mat()

    def __load_mat(self):

        """
        Add results from KRAKEN simulation as class attributes.
        TODO: Parametarize which key-value pair we want to extract
        """
        # Load KRAKEN simulation data
        mat_calls = loadmat(self.__path)

        # Extract individual vectors
        self.p_t_noise = mat_calls['p_t_r_noise']
        self.t_dec_min = np.squeeze(mat_calls['t_dec_min'])
        self.t_dec_max = np.squeeze(mat_calls['t_dec_max'])
        self.labels = np.squeeze(mat_calls['labels']).astype('float32')
        self.fs = float(np.squeeze(mat_calls['fs']))
        self.T = float(np.squeeze(mat_calls['T']))

        # # Load noise only data if possible
        if self.__noise_only_path is not None:
            mat_noise = loadmat(self.__noise_only_path)
            self.p_t_noise = np.concatenate((self.p_t_noise, mat_noise['p_t_noise_only']), axis=1)
            self.labels = np.concatenate((self.labels, np.squeeze(mat_noise['labels_noise_only']).astype('float64')), axis=1)


        # Number of examples and labels
        self.n_examples = self.labels.shape[1]
        self.n_samples_per_example = self.labels.shape[0]

    def normalize_wav(self):
        self.p_t_noise = (self.p_t_noise - self.p_t_noise.mean(axis=0)) / self.p_t_noise.std(axis=0)


    def create_spectrograms(self, fs=None, rand_shift=False, nperseg=31, noverlap=23, nfft=500, verbose=False, channels=1, resize=None):
        
        """
        Create spectrograms using dispersive calls from KRAKEN simulation. 
        TODO: Specify number of files or max file memory.

        Parameters
        ----------
        fs: float, sampling frequencey used in KRAKEN simulation.
        rand_shift: bool, whether or not to apply a random wrapping shift to the spectrogram.
        nperseg: int, number of samples per window.
        noverlap: int, number of samples overlap between windows.
        nfft: int, number of frequency 'bins' on the frequency axis of the spectrogram.
        verbose: bool, print example number as spectrograms are calculated.
        channels: int, number of channels per example. If it is > 1, then the noverlap values from
                  the variable noverlap_list within the function are used.
        resize: tuple, height and width of the desired shape to resize the raw spectrogram size to.

        Returns
        -------
        X: array-like, calculated spectrograms of shape (n_examples, Zxx.shape[0], Zxx.shape[1]).
        y: array-like, labels of shape (n_labels, n_examples).
        """

        # For 3 channel nperseg = 60, noverlap = [59, 40, 20], nfft = 447
        # For 1 channel nperseg = 60, noverlap = 52, nfft = 750

        if fs is None:
            fs = self.fs

        noverlap_list = [nperseg-1, (2/3)*nperseg, (1/3)*nperseg]

        if channels > len(noverlap_list):
            raise ValueError("Number of channels desired is greater than specified noverlaps in the function's 'noverlap_list' variable.")

        if resize is None:
            # Generate one spectrogram to get the dimensions for the current settings
            [_,_,Zxx] = stft(x=self.p_t_noise[:,0], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            self.X = np.zeros((self.n_examples, channels, Zxx.shape[0], Zxx.shape[1]), dtype='float32')
        else:
            self.X = np.zeros((self.n_examples, channels, resize[0], resize[1]), dtype='float32')
        self.y = self.labels

        for sig in tqdm(range(self.n_examples), disable=not verbose):

            # Random wrapping using start/end time of call or just length of data if it is noise
            if rand_shift and self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig], self.t_dec_min[sig], self.t_dec_max[sig])
            elif rand_shift and not self.labels[4,sig]:
                self.p_t_noise[:,sig] = self.__wrap_signal_random(self.p_t_noise[:,sig])

            if channels == 1:
                [f, t, Zxx] = stft(x=self.p_t_noise[:,sig], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                log_spec = np.flipud(10*np.log10(np.abs(Zxx)**2))
                if resize is not None:
                    log_spec = resize(log_spec, resize, anti_aliasing=True)
                
                # Add channel dimension
                log_spec = log_spec[np.newaxis, :,:]
                self.X[sig,:,:,:] = log_spec

            elif channels > 1:
                for c in range(channels):
                    [f, t, Zxx] = stft(x=self.p_t_noise[:,sig], fs=fs, nperseg=nperseg, noverlap=noverlap_list[c], nfft=nfft)
                    log_spec = np.flipud(10*np.log10(np.abs(Zxx)**2))
                    if resize is not None:
                        log_spec = resize(log_spec, resize, anti_aliasing=True)
                    
                    # Add channel dimension
                    log_spec = log_spec[np.newaxis, :,:]
                    self.X[sig,c,:,:] = log_spec
            
            else:
                raise ValueError("Channels must be between 1 and {}".format(len(channels)))

        # Normalize between 0 and 1
        self.X = (self.X - self.X.min(axis=(2,3), keepdims=True)) / (self.X.max(axis=(2,3), keepdims=True) - self.X.min(axis=(2,3), keepdims=True))

        # Make first dimension the number of examples
        self.y = self.y.T.astype('float32')

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

    def save_split_h5(self, train_path, val_path, test_path=None, val_size=None, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
        
        """
        Save spectrograms and labels in h5 format.
        Parameters
        ----------
        path: str, path where the h5 file will be saved.
        """
        
        if self.X is None or self.y is None:
            raise Exception("Spectram data has not yet been created.")

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        
        # Test split from training data
        if test_size is not None:
            test_size = test_size / train_size
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

            with h5py.File(test_path, 'w') as f:
                f.create_dataset("data", data=X_test)
                f.create_dataset("labels", data=y_test)

        with h5py.File(train_path, 'w') as f:
            f.create_dataset("data", data=X_train)
            f.create_dataset("labels", data=y_train)

        with h5py.File(val_path, 'w') as f:
            f.create_dataset("data", data=X_val)
            f.create_dataset("labels", data=y_val)