import h5py

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