import os
import argparse
import csv

import torch
import numpy as np
from scipy.io import wavfile
from resampy import resample
from tqdm import tqdm
import warnings

from models.tcn import BranchedTCN_CE
import datasets.dataLoader as dataLoader
import config

# Input options
parser = argparse.ArgumentParser(description="Apply the gunshot detection/ranging TCN to 'wav' acoustic data.")
parser.add_argument('--acoustic_data', type=str,
                    help="directory where the 'wav' files to be scanned are stored")
parser.add_argument('--save_dir', type=str,
                    help="directory where the extracted gunshot detections should be stored.")
parser.add_argument('--batch_size', type=int, default=128,
                    help="Number of extracted windows to process with the TCN simultaneously")
parser.add_argument('--verbose', action='store_true',
                    help='display progress for each file scan (default: False)')
args = parser.parse_args()

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

# Load model and weights
model = BranchedTCN_CE(input_size=232, output_size=2, num_channels=[368]*9, kernel_size=6, dropout=0.35, softmax=True).to(device)
model.load_state_dict(torch.load(os.path.join(config.models_dir + '/L2NormFixLoss2', 'weights_100.pt'), map_location=device)['model_state_dict'])

# Put model in eval mode
model.eval()

# Make model parallel
model = torch.nn.DataParallel(model)

class Scanner():
    
    def __init__(self):

        # Counter
        self.sig = 1

        # Initialize csv
        column_name = ["File", "Start", "End"]
        with open(os.path.join(args.save_dir, 'results.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(column_name)

    def scan_wav(self, new_wav_path, save_path, fs_target, T_target, batch_size):
        
        # Get audiofile
        fs0, s0 = wavfile.read(new_wav_path)

        # Get samples / window 
        n_samples = round(fs0 * T_target)
        n_samples_dec = round(fs_target * T_target)
        
        # Calculate number of batches (this ignores leftover)
        n_batches = len(s0) // (batch_size * n_samples)
        
        # Check we have at least one batch
        if n_batches == 0:
            warnings.warn("Specified batch size exceeds the length of the wav file for {}".format(new_wav_path.split('/')[-1]))
            return
            
        # Go through the file as long as we have an entire batch left
        curr_sample = 0
        
        # Get image transforms
        transforms = dataLoader.get_image_transforms()['eval']

        # print file name and start scanning
        print('\n' + 'file: ' + new_wav_path.split('/')[-1])
        for batch in tqdm(range(n_batches), disable= not args.verbose):
            
            # Array to store a batch of signals
            S = np.zeros((batch_size, n_samples_dec))

            # Save indices
            I = np.zeros((batch_size,3),dtype=object)
            
            # Collect data and calculate spectrograms
            for i in range(batch_size):
                
                # Get a window
                s_curr = s0[curr_sample:curr_sample+n_samples]

                # Decimate signal and adjust fs
                s_curr = resample(s_curr, fs0, fs_target)

                # Mean center and L2 norm divide
                s_curr = s_curr - s_curr.mean()
                s_curr = s_curr / np.sqrt(np.sum(s_curr ** 2))
                
                # save window and indices
                S[i,:] = s_curr
                I[i,0] = new_wav_path
                I[i,1] = curr_sample / fs0
                I[i,2] = (curr_sample + n_samples) / fs0
                
                # Iterate samples
                curr_sample += n_samples

            # Calculate spectrograms
            X = dataLoader.to_spect(S).squeeze(axis=1).copy()

            # Predict current batch
            X_torch = transforms(torch.from_numpy(X).float()).squeeze().to(device)
            outputs = model(X_torch)
            
            # Get detected signals
            S = S[outputs[:,2].cpu().squeeze() >= 0.5,:]
            I = I[outputs[:,2].cpu().squeeze() >= 0.5,:]
            ranges = config.max_range*outputs[outputs[:,2].cpu().squeeze() >= 0.5,0].cpu()
            
            # Save detected signals
            if len(S) > 0:
                for wav, r in zip(S, ranges):
                    r = str(int(r/100) / 10).replace('.','_')
                    wavfile.write(os.path.join(save_path,str(self.sig) + '-'+ r + '|1' + '.wav'), fs_target, wav)
                    self.sig += 1
                
                with open(os.path.join(args.save_dir, 'results.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(I)

if __name__== "__main__":
    
    # Collect wav files in directory
    wav_files = [f for f in os.listdir(args.acoustic_data) if os.path.isfile(os.path.join(args.acoustic_data,f)) and ".wav" in f]
    scanner = Scanner()
    
    # Call function
    fs_target = config.fs
    T_target = config.T
    for file_idx in range(len(wav_files)):
        scanner.scan_wav(os.path.join(args.acoustic_data, wav_files[file_idx]), args.save_dir, fs_target, T_target, args.batch_size)