import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
        
class FrequencyBandZeroing:

    """
    Class to be used in transforms.Compose() to randomly zero out frequencies.
    """

    def  __init__(self, max_freq_width = 30, max_t_width=30, num_f=2, num_t=4):
        self.max_freq_width = max_freq_width
        self.max_t_width = max_t_width
        self.num_f = num_f
        self.num_t = num_t

    def freq_band_zeroing(self, x):

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


        zero_width_f = torch.randint(0, self.max_freq_width, size=(self.num_f,))
        offset_f = torch.randint(0, H - zero_width_f.max(), size=(self.num_f,))

        if self.num_t:
            zero_width_t = torch.randint(0, self.max_t_width, size=(self.num_t,))
            offset_t = torch.randint(0, W - zero_width_t.max(), size=(self.num_t,))
        
        for wf, of in zip(zero_width_f, offset_f):
            x[:,of:(of + wf),:] = 0
        
        if self.num_t:
            for wf, of in zip(zero_width_t, offset_t):
                x[:,:,of:(of + wf)] = 0

        return x

    def __call__(self, tensor):
        return self.freq_band_zeroing(tensor)

class Normalize1DChannel:

    def __init__(self, mu_list, std_list):
        self.mu_list = torch.tensor(mu_list, dtype=torch.float32)
        self.std_list = torch.tensor(std_list, dtype=torch.float32)

    def norm(self, x):
        return (x - self.mu_list.view(1,-1,1)) / self.std_list.view(1,-1,1)

    def __call__(self, tensor):
        return self.norm(tensor)

class ZeroOneNorm:

    def __init__(self):
        pass

    def norm(self, x):
        return (x - x.min(axis=2).values.view(1,-1,1)) / (x.max(axis=2).values - x.min(axis=2).values).view(1,-1,1)

    def __call__(self, tensor):
        return self.norm(tensor)

if __name__ == "__main__":
    z = Normalize1DChannel([2,4], [3,2])
    a = torch.tensor([[1,2,3],[6,2,9]])
    print(z(a.unsqueeze(0)))
