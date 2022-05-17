##################################################################
#                            Paths                               #
##################################################################

# Root data directory
data_dir = '/workspace/data'

# Where model checkpoints are saved
models_dir = data_dir + '/models'

# Data from MATLAB simulation
mat_data = data_dir + '/grid_data_atten.mat'
mat_noise = data_dir + '/noise_only_data_atten.mat'

# Data for model development
train_data = data_dir + '/grid_data_atten_big_train.h5'
val_data = data_dir + '/grid_data_atten_big_train.h5'
test_data = data_dir + '/grid_data_atten_big_test.h5'

# Collection of gunshot wav files that have been warped
warped_dir = data_dir + '/new_warped_calls_correct_depths_D70'
# Location to store WAV files of gunshots detected by the network
collected_data = data_dir + '/call_collection'
# Spectrograms and labels of warped experimental data 
exp_data = data_dir + 'wav_calls.h5'

##################################################################
#                         Constants                              #
##################################################################

# Maximum range of training data
max_range = 35000 # [km]
# Training data mean
mu = -30.1996 # [dB]
# Training data std
std = 11.9025 # [dB]
# Trained fs
fs = 600 # [samples / s]
# Trained signal window period
T = 5 # [s]