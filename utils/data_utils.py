#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import sys
sys.path.append('../')
import numpy as np
import scipy.signal as signal
import torch
import pickle as pkl
import os.path as osp
from torch.utils import data
from utils.training_utils import set_seeds

# General data utilities.

class Dataset(data.Dataset):
    def __init__(self, config, chopped_spikes, session_names):
        super().__init__()
        device = torch.device('cuda:0')

        self.session_names = session_names
        self.n_samples = chopped_spikes.shape[0]
        self.n_channels = chopped_spikes.shape[-1]

        self.n_heldout = int(config.train.pct_heldout * self.n_channels)
        self.n_heldin = self.n_channels - self.n_heldout

        self.has_heldout = self.n_heldout > 0
        heldin_channels = get_heldin_mask(config, self.n_channels)
        
        self.heldin_spikes = chopped_spikes[..., heldin_channels].to(device)
        self.heldout_spikes = chopped_spikes[..., ~heldin_channels].to(device)

        self.max_train_spks = self.heldin_spikes.max().item() + 3

    def __len__(self):
        return self.n_samples

    

    def __getitem__(self, index):
        heldout = self.heldout_spikes[index] if self.has_heldout else []
        return self.heldin_spikes[index], heldout, self.session_names[index]

    def get_dataloader(self, generator, shuffle=True):
        '''
        Args:
        Returns:
        '''
        return data.DataLoader(
            self,
            batch_size=self.config.train.batch_size,
            generator=generator,
            shuffle=shuffle
        )

def get_dataloaders(config, chopped_spikes, session_names):
    ''' TODO
    '''
    # load into generic training dataset
    all_data = Dataset(config, chopped_spikes, session_names)

    def _init_fn(worker_id):
        set_seeds(config)

    generator = torch.Generator()
    generator.manual_seed(config.train.val_seed)

    shuffled_indicies = torch.randperm(all_data.n_samples, generator=generator)
    split_index = int(config.train.pct_val * all_data.n_samples)
    train_indicies = list(shuffled_indicies[split_index:])
    val_indicies = list(shuffled_indicies[:split_index])

    print(f'\nTraining set samples: {len(train_indicies)}\nValidation set samples: {len(val_indicies)}')

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(all_data, train_indicies),
        batch_size=config.train.batch_size,
        worker_init_fn=_init_fn,
        generator=generator,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(all_data, val_indicies),
        batch_size=config.train.batch_size,
        worker_init_fn=_init_fn,
        generator=generator,
        shuffle=False,
    )

    with open(osp.join(config.dirs.save_dir, 'dataset.pkl'), 'wb') as dfile:
        pkl.dump(all_data, dfile)

    return train_dataloader, val_dataloader, all_data


def get_heldin_mask(config, n_channels):
    '''
    '''
    n_heldout = int(config.train.pct_heldout * n_channels)

    np.random.seed(config.train.heldout_seed)
    heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)

    heldin_channels_mask = torch.ones(n_channels, dtype=bool)
    heldin_channels_mask[heldout_channels] = False

    return heldin_channels_mask
    

def chop(data, seq_len, overlap):
    ''' TODO
    '''
    shape = (int((data.shape[0] - overlap) / (seq_len - overlap)), seq_len, data.shape[-1])
    strides = (data.strides[0] * (seq_len - overlap), data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape, strides).copy().astype('f')


def smooth(data, gauss_width, bin_width, causal):
    '''
    '''
    kern_sd = int(gauss_width / bin_width)
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)

    if causal: 
        window[len(window) // 2:] = 0
    
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    
    return np.apply_along_axis(filt, 0, data)

