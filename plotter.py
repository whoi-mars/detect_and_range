import matplotlib.pyplot as plt
import numpy as np
import random

def plot_grid(images_arr, labels_arr, label_text, title, dim=(3, 3), randomize=True):
    
    """
    Plots an array of images and their corresponding labels.
    
    Parametersnperseg: int, number of samples per window.
        noverlap: int, number of samples overlap between windows.
        nfft: int, number of frequency 'bins' on the frequency axis of the spectrogram.
    ----------
    images_arr: 2D array-like, array of images
    labels_arr: array-like, array of labels which correspond in order to the images
    dim: tuple, dimensions of the plot in terms of number of images
    randomize: bool, whether or not to take a random subset of the images of size dim[0]*dim[1]. In
               this case, there must be at least dim[0] * dim[1] many images.
    """

    if type(dim) != tuple or len(dim) != 2:
        raise Exception("dim needs to be a 2D tuple specifying the dimensions of the image grid")

    if randomize:
        total_examples = dim[0] * dim[1]
        inds = random.choices(range(len(images_arr)), k=total_examples)
        images_arr = images_arr[inds,:,:,:]
        labels_arr = labels_arr[inds]

    if dim == (1, 1):
        plt.title(title)
        plt.imshow(np.squeeze(images_arr[0]), aspect='auto')
        plt.axis('off')
        plt.title("{}: {}".format(label_text, round(labels_arr[0],2)))
        plt.show()
        return

    fig, axes = plt.subplots(dim[0], dim[1], figsize=(6, 6))
    plt.suptitle(title)
    axes = axes.flatten()
    i = 0
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.squeeze(img), aspect='auto')
        ax.axis('off')
        ax.set_title("{}: {}".format(label_text, round(labels_arr[i],2)))
        i += 1
    plt.tight_layout()
    plt.show()