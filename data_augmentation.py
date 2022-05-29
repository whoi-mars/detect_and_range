import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def freq_band_zeroing(x, max_freq_width = 30):

    """
    Function to randomly zero-out a band of frequencies

    Parameters
    ----------
    x: array-like, input spectrogram.
    max_freq_width: int, maximum continuous bandwidth to zero.

    Returns
    -------
    x: array-like, spectrogram with zeroed-out frequencies
    """

    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        C, H, W = x.shape
    elif len(x.shape) == 3:
        C, H, W = x.shape
    else:
        raise ValueError(f"x must be shape (H, W) or (C, H, W), is now {len(x.shape)}")

    for c in range(C):
        zero_width = torch.randint(0, max_freq_width, size=(1,))
        offset = torch.randint(0, H - zero_width.item(), size=(1,))

        x[c,offset:(offset + zero_width),:] = 0

    return x
        
class FrequencyBandZeroing:

    """
    Class to be used in transforms.Compose() to randomly zero out frequencies.
    """

    def  __init__(self, max_freq_width = 30):
        self.max_freq_width = max_freq_width

    def __call__(self, tensor):
        return freq_band_zeroing(tensor, self.max_freq_width)


if __name__ == "__main__":
    from dataLoader import load_h5
    X, y = load_h5("./data/grid_data_atten_big_test.h5")
    idx = np.random.randint(X.shape[0])
    x = torch.from_numpy(X[idx])

    Ts = transforms.Compose([
        FrequencyBandZeroing()
    ])

    x_zeroed = Ts(torch.clone(x))

    fig, axes = plt.subplots(nrows=2, ncols=3)

    data = [x, x_zeroed]
    titles = ["Original image", "Zeroed out image"]
    for x, title, axs in zip(data, titles, axes):
        for k, (xx, ax) in enumerate(zip(x, axs)):
            ax.imshow(xx)
            if k == 1:
                ax.set_title(title)

    plt.show()