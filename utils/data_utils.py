#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import sys
sys.path.append('../')
import numpy as np
import scipy.signal as signal
import torch
from utils.training_utils import set_seeds
import pickle as pkl
import os.path as osp

# General data utilities.
from dataset import Dataset

def get_dataloaders(config, chopped_spikes, session_names):
    ''' TODO
    '''
    # load into generic training dataset
    all_data = Dataset(config, chopped_spikes, session_names)

    generator = torch.Generator()
    generator.manual_seed(config.train.val_seed)

    shuffled_indicies = torch.randperm(all_data.n_samples, generator=generator)
    split_index = int(config.train.pct_val * all_data.n_samples)
    train_indicies = list(shuffled_indicies[split_index:])
    val_indicies = list(shuffled_indicies[:split_index])

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(all_data, train_indicies),
        batch_size=config.train.batch_size,
        generator=generator,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(all_data, val_indicies),
        batch_size=config.train.batch_size,
        generator=generator,
        shuffle=True,
    )

    with open(osp.join(config.dirs.save_dir, 'dataset.pkl'), 'wb') as dfile:
        pkl.dump(all_data, dfile)

    return train_dataloader, val_dataloader, all_data
    

def chop(data, seq_len, overlap):
    ''' TODO
    '''
    shape = (int((data.shape[0] - overlap) / (seq_len - overlap)), seq_len, data.shape[-1])
    strides = (data.strides[0] * (seq_len - overlap), data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape, strides).copy().astype('f')

def smooth_spikes(data, gauss_width, bin_width, causal):
    kern_sd = int(gauss_width / bin_width)
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    if causal: 
        window[len(window) // 2:] = 0
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    return np.apply_along_axis(filt, 0, data)
