# Detect and Range

Detect and Range is a PyTorch package for developing deep learning models that detect underwater acoustic signals and estimate source range from passive acoustic recordings.

The repository was developed for research in passive acoustic monitoring and underwater acoustics, with an emphasis on training Temporal Convolutional Networks (TCNs) using synthetic data generated with the KRAKEN acoustic propagation model and evaluating their performance on experimental recordings.

The package includes tools for

- loading acoustic datasets stored in HDF5 format,
- generating spectrograms from time-domain recordings,
- training TCN-based neural networks,
- evaluating trained models, and
- preparing new datasets from simulations and experimental recordings.

---

## Features

- Temporal Convolutional Network (TCN) architectures for acoustic detection and ranging
- Multi-task learning with joint detection and range prediction
- Automatic spectrogram generation from waveform data
- HDF5-based dataset interface
- Data augmentation utilities for spectrograms
- Training checkpoint support
- Optional Weights & Biases experiment logging
- Utilities for creating datasets and scanning WAV files
- Example notebooks for data generation and model evaluation

---

## Repository Structure

```
detect_and_range/
├── datasets/
│   Dataset classes and PyTorch dataloaders
│
├── losses/
│   Custom loss functions used during training
│
├── models/
│   Temporal Convolutional Network implementations
│
├── scripts/
│   Stand-alone utility scripts
│
├── utils/
│   Data preparation, augmentation, and visualization utilities
│
├── config.py
│   Dataset locations and preprocessing constants
│
├── train_tcn.py
│   Main training script
│
├── create_data.ipynb
│   Example notebook for generating datasets
│
├── evaluate.ipynb
│   Example notebook for evaluating trained models
│
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/whoi-mars/detect_and_range.git
cd detect_and_range
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Data

Training and evaluation data are stored as HDF5 files.

Dataset locations are specified in `config.py`, including paths for

- training data
- validation data
- testing data
- experimental recordings
- MATLAB simulation data
- model checkpoints

Before training, update these paths to match your local directory structure.

---

## Quick Start

### 1. Configure data locations

Edit `config.py` and set the paths to your datasets and model directory.

### 2. Train a model

The training script exposes a number of command-line arguments for controlling model architecture and optimization, including

- batch size
- learning rate
- dropout
- number of TCN levels
- kernel size
- checkpoint directory
- random seed
- high-pass filtering
- optional Weights & Biases logging

For example,

```bash
python train_tcn.py \
    --batch_size 128 \
    --end_epoch 50 \
    --lr 1e-3 \
    --checkpoint_dir experiment_01
```

### 3. Evaluate the trained model

Example evaluation workflows are provided in `evaluate.ipynb`.

---

## Models

The primary model implementation is based on **Temporal Convolutional Networks (TCNs)**.

The repository includes branched TCN architectures that jointly predict

- acoustic signal detection
- source range

using a shared temporal feature extractor followed by task-specific prediction heads.

---

## Utilities

The repository includes several utilities supporting data preparation and analysis.

### Dataset generation

`create_data.ipynb`

Example workflow for creating HDF5 datasets suitable for model training.

### WAV scanning

`scripts/scan_wav.py`

Utility for scanning collections of WAV recordings and extracting candidate detections.

### Data augmentation

`utils/data_augmentation.py`

Spectrogram augmentation routines used during training.

---

## Configuration

Most project-specific settings are centralized in `config.py`, including

- dataset paths
- model checkpoint locations
- STFT parameters
- normalization statistics
- training constants

This allows the training and evaluation scripts to share a consistent configuration.

---

## Requirements

See `requirements.txt` for the complete list of dependencies.

---

## Citation

If you use this repository in your research, please cite the following.

```bibtex
@article{goldwater2023,
    author = {Goldwater, Mark and Zitterbart, Daniel P. and Wright, Dana and Bonnel, Julien},
    title = {Machine-learning-based simultaneous detection and ranging of impulsive baleen whale vocalizations using a single hydrophone},
    journal = {The Journal of the Acoustical Society of America},
    volume = {153},
    number = {2},
    pages = {1094-1107},
    year = {2023},
    month = {02},
}
```
