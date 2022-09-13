#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import random
import logging
import requests
import os.path as osp
#────#
import h5py
import torch
import numpy as np
import scipy.signal as signal
from tqdm.auto import tqdm
from torch.utils import data
from nlb_tools import make_tensors
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    save_to_h5,
    combine_h5,
    h5_to_dict,
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors)
from utils import set_seeds

import warnings; warnings.simplefilter('ignore')
'''────────────────────────────── datasets.py ───────────────────────────────'''
# This file contains the class Dataset and the methods for downloading the
# datasets.

'''
   ─────────────────────── DANDI API DATASET STRINGS ────────────────────────
'''
# Hide NaN warnings for test data from nlb_tools
logging.basicConfig(level=logging.ERROR)

# Each dataset has a unique ID that allows us to directly download w/o DandiCLI
api_url = ('https://api.dandiarchive.org/api/assets/', '/download/')
id_dict = dict(
    mc_maze = dict(
        mc_maze_train = '26e85f09-39b7-480f-b337-278a8f034007',
        mc_maze_test = '1bd112a4-5ec5-4033-ac30-d88e70e993d9'),
    mc_rtt = dict(
        mc_rtt_train = '2ae6bf3c-788b-4ece-8c01-4b4a5680b25b',
        mc_rtt_test = '648a7418-98e8-4413-ba97-3772dd325ecc'),
    dmfc_rsg = dict(
        dmfc_rsg_train = '5e92e8db-5212-4607-8b90-3f5a5e319537',
        dmfc_rsg_test = '6718e18a-944f-40f2-8358-d5ff562fb0a0'),
    area2_bump = dict(
        area2_bump_train = 'ded26b6c-418d-43f5-8a37-dfd072c2dbd4',
        area2_bump_test = 'f907b288-a9b4-4c6e-9d66-6812f306e253'),
    mc_maze_small = dict(
        mc_maze_small_train = '7821971e-c6a4-4568-8773-1bfa205c13f8',
        mc_maze_small_test = '544a660e-a173-46f3-985a-60fd67e6ae1f'),
    mc_maze_medium = dict(
        mc_maze_medium_train = '7ef450a8-8684-42e2-8598-cd38ca2b2e50',
        mc_maze_medium_test = 'ac554f8e-7008-4030-9640-6f1cf838a23c'),
    mc_maze_large = dict(
        mc_maze_large_train = 'e67b57b2-e9ad-4d95-b9e3-1262997360dc',
        mc_maze_large_test = 'f42a7bd1-71f4-4972-84a7-2cd1dfbed8a1'))
'''
────────────────────────────────────────────────────────────────────────────────
'''

def get_dataloaders(config, mode):
    ''' Creates Dataset objects and creates the DataLoaders from them.

    Args:
        config (dict): A config object.
        mode (str): ['original', 'random', 'test', 'none'] Which validation set
                    shold be used. Original is the nlb given validation set,
                    random is a random subset of the combined train and
                    validation set, none is no validation set
    Returns:
        train_dataloader (DataLoader): The training set DataLoader.
        (test/val)_dataloader (DataLoader, None): The test or validation set
                                                  DataLoader. In 'none' this is
                                                  None.
    '''
    data_path = f'{config["setup"]["data_dir"]}{config["setup"]["dataset"]}'
    data_path += f'_{config["train"]["seq_len"]}_{config["train"]["overlap"]}'
    data_path += f'_{config["train"]["lag"]}.h5'

    generator = torch.Generator()
    generator.manual_seed(config['setup']['seed'])

    def _init_fn(worker_id):
        set_seeds(config)
    
    if mode == 'train':
        # train_data = Dataset(config, data_path, 'train')
        # train_dataloader = train_data.get_dataloader(generator, shuffle=True)
        # return train_dataloader
        print('test')

    elif mode == 'train_val':
        train_data = Dataset(config, data_path, 'train')

        n_samples = len(train_data)
        split_index = int(0.2 * n_samples)
        shuffled_indicies = torch.randperm(n_samples, generator=generator)

        train_indicies = list(shuffled_indicies[:split_index])
        val_indicies = list(shuffled_indicies[split_index:])

        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, train_indicies),
            batch_size=config['train']['batch_size'],
            generator=generator,
            worker_init_fn=_init_fn,
            shuffle=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, val_indicies),
            batch_size=config['train']['batch_size'],
            shuffle=False
        )

        return train_dataloader, val_dataloader
    
    # elif mode == 'test':
    #     test_data = Dataset(config, data_path, 'test')

    #     test_dataloader = test_data.get_dataloader(generator, shuffle=False)
    #     return test_dataloader

    # if mode == 'cross-val':
    #     train_data = Dataset(config, data_path, 'train')
    #     val_data = Dataset(config, data_path, 'val')

    #     val_data.clip_val(train_data.spikes_heldin.max().item() + 3)
    #     train_dataloader = train_data.get_dataloader(generator, shuffle=True)
    #     val_dataloader = val_data.get_dataloader(generator, shuffle=False)
    #     return train_dataloader, val_dataloader

    # elif mode == 'random':
    #     trainval_data = Dataset(config, data_path, 'trainval')

    #     n_samples = len(trainval_data)
    #     shuf_gen = torch.Generator()
    #     shuf_gen.manual_seed(config['setup']['seed'])
    #     shuffled_inds = torch.randperm(n_samples, generator=shuf_gen)

    #     split_ind = int((1 - config['train']['val_ratio']) * n_samples)
    #     train_inds = list(shuffled_inds[:split_ind])
    #     val_inds = list(shuffled_inds[split_ind:])

    #     train_dataloader = torch.utils.data.DataLoader(
    #         torch.utils.data.Subset(trainval_data, train_inds),
    #         batch_size=config['train']['batch_size'],
    #         generator=generator,
    #         worker_init_fn=_init_fn,
    #         shuffle=True)
    #     val_dataloader = torch.utils.data.DataLoader(
    #         torch.utils.data.Subset(trainval_data, val_inds),
    #         batch_size=config['train']['batch_size'],
    #         generator=generator,
    #         shuffle=False)
    #     return train_dataloader, val_dataloader



def chop_data(data, chopsize, overlap, lag_bins):
    ''' Chops data trial by trail (or segment by segment) into overlapping segments.'''
    chopped_data = []
    for trial in data:
        if lag_bins > 0:
            trial = trial[:-lag_bins, :]
        shape = (
            int((trial.shape[0] - overlap) / (chopsize - overlap)),
            chopsize,
            trial.shape[-1],
        )
        strides = (
            trial.strides[0] * (chopsize - overlap),
            trial.strides[0],
            trial.strides[1],
        )

        chopped_trial = np.lib.stride_tricks.as_strided(trial, shape=shape, strides=strides).copy().astype('f')
        chopped_data.append(chopped_trial)

    chopped_data = np.array(chopped_data)

    return chopped_data.reshape((
        chopped_data.shape[0] * chopped_data.shape[1], 
        chopped_data.shape[2], 
        chopped_data.shape[3]
    ))


class Dataset(data.Dataset):
    def __init__(self, config, filename, mode):
        '''init Dataset

        Args:
            config (dict): A config object.
            filename (str): path to the dataset.h5 file.
            mode (str): ['test', 'train' , 'val', 'trainval'] The dataset type.
        '''
        super().__init__()
        
        with h5py.File(filename, 'r') as h5file:
            h5dict = h5_to_dict(h5file)
            self.config = config
            self.mode = mode

            device = torch.device('cuda:0')

            def set_sizes(self):
                ''' Helper function that assigns the number of samples, trial
                length, forward pass length, number of heldin neurons, and
                number of heldout neurons.
                '''
                self.n_samples = self.spikes_heldin.shape[0]
                self.n_heldin = self.spikes_heldin.shape[2]
                self.n_heldout = self.spikes_heldout.shape[2]
                self.n_neurons = self.n_heldin + self.n_heldout
                
                self.max_train_spks = h5dict['train_spikes_heldin'].max().item() + 3

            if mode == 'train':
                self.spikes_heldin = torch.Tensor(h5dict['train_spikes_heldin'].astype(np.float32)).to(device)
                self.spikes_heldout = torch.Tensor(h5dict['train_spikes_heldout'].astype(np.float32)).to(device)

                set_sizes(self)

            elif mode == 'test':
                self.spikes_heldin = torch.Tensor(h5dict['test_spikes_heldin'].astype(np.float32)).to(device)
                self.spikes_heldout = torch.Tensor(h5dict['test_spikes_heldout'].astype(np.float32)).to(device)

                set_sizes(self)

    def __len__(self):
        '''
        Returns:
            int: The number of samples or trials.
        '''
        return self.spikes_heldin.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index into the batches.
        Returns:
            spikes_heldin (np.ndarray): Spikes held-in.
            spikes_heldout (np.ndarray): Spikes held-out.
            spikes_all_fp (np.ndarray): Spikes forward pass.
        '''
        return (
            self.spikes_heldin[index],
            self.spikes_heldout[index],
        ) 

    def clip_max(self, max, indicies):
        self.spikes_heldin[indicies] = torch.clamp(self.spikes_heldin[indicies], max=max)

    def get_dataloader(self, generator, shuffle=True):
        return data.DataLoader(self,
            batch_size=(
                self.config['train']['batch_size'] if self.mode == 'train' else 4096
            ),
            generator=generator,
            shuffle=shuffle)

def verify_dataset(config):
    '''Checks if the correct dataset.h5 file is downloaded, if it isn't then it
    prompts the user if they would like to download it.

    Args:
        config (dict): A config object.
    '''
    path = f'{config["setup"]["data_dir"]}{config["setup"]["dataset"]}'
    path += f'_{config["train"]["seq_len"]}_{config["train"]["overlap"]}'
    path += f'_{config["train"]["lag"]}.h5'

    if os.path.isfile(path):
        return None
    else:
        print(f'Dataset could not be found at: {path}')
        response = input('Would you like to create it? (y/n): ')
        while response != 'y' and response != 'n':
            response = input("Please enter 'y' or 'n': ")
        if response == 'n': exit()

        if config['setup']['dataset'] == 'mc_rtt':
            download_mc_rtt(config['setup']['data_dir'])

        create_h5_data(
            config["setup"]["data_dir"], 
            config["setup"]["dataset"],
            config["train"]["seq_len"],
            config["train"]["overlap"],
            config["train"]["lag"]
        )


def download_mc_rtt(data_dir):
    ''' Download datasets, combine train/test & prep data, delete old files.

    Args:
        path (str): The path
    '''

    filepath = f'{data_dir}mc_rtt.nwb'

    if not osp.isfile(filepath):
        print('\nDownloading mc_rtt.nwb')

        # Download a specific dataset to avoid the use of Dandi CLI
        url = 'https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/'
        response = requests.get(url, stream=True)
        
        # Get dataset file size
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        
        # Make progress bar
        progress_bar_file = tqdm(
            total=total_size_in_bytes, unit='iB',
            unit_scale=True, leave=False,
            # desc=dataset_type,
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt} / {total_fmt}'
            )

        # Start download and update progress bars
        with open(filepath, 'wb') as file:
            for data in response.iter_content(1024): #1024
                progress_bar_file.update(len(data))
                file.write(data)

        # Close file progress bar once file downloaded
        progress_bar_file.close()

def seg_arr(data):
    ''' Segments up a continous datastream split by Nans
    '''
    # shape of data: T (time) x C (channels)
    tmp_list = []
    for channel in data:
        tmp_list.append([channel[seg] for seg in np.ma.clump_unmasked(np.ma.masked_invalid(channel))])

    return np.transpose(np.array(tmp_list), (1, 2, 0))


def create_h5_data(data_dir, dataset, seq_len, overlap, lag):
    ''' Turn train/test '.nwb' files into a single .h5 file and save
    '''
    filepath = f'{data_dir}{dataset}.nwb'

    dataset_obj = NWBDataset(filepath) # NWB Object
    dataset_obj.resample(10)

    vel_trans = dataset_obj.data.finger_vel.to_numpy().T # idx 0 is x; idx 1 is y
    hi_spikes_trans = dataset_obj.data.spikes.to_numpy().T
    ho_spikes_trans = dataset_obj.data.heldout_spikes.to_numpy().T

    vel_segments = seg_arr(vel_trans) # (4, 16220, 2)
    hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
    ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

    lag_bins = int(round(lag / dataset_obj.bin_width))
    lagged_vel_segments = np.array([seg[lag_bins:] for seg in vel_segments])
    train_vel_segments, test_vel_segments = lagged_vel_segments[:3], np.expand_dims(lagged_vel_segments[3], 0)

    train_vel_segments = chop_data(train_vel_segments, seq_len, overlap, lag_bins=0)[:, -1, :]
    test_vel_segments = chop_data(test_vel_segments, seq_len, overlap, lag_bins=0)[:, -1, :]

    train_hi_segments = chop_data(hi_spike_segments[:3], seq_len, overlap, lag_bins) # (8097, 30, 98)
    train_ho_segments = chop_data(ho_spike_segments[:3], seq_len, overlap, lag_bins) # (8097, 30, 32)

    test_hi_segments = chop_data(np.expand_dims(hi_spike_segments[3], 0), seq_len, overlap, lag_bins) # (16191, 30, 98)
    test_ho_segments = chop_data(np.expand_dims(ho_spike_segments[3], 0), seq_len, overlap, lag_bins) # (16191, 30, 32)

    kern_sd = int(round(30 / dataset_obj.bin_width))
    window_30ms = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window_30ms /= np.sum(window_30ms)
    filt_30 = lambda x: np.convolve(x, window_30ms, 'same')

    kern_sd = int(round(50 / dataset_obj.bin_width))
    window_50ms = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window_50ms /= np.sum(window_50ms)
    filt_50 = lambda x: np.convolve(x, window_50ms, 'same')

    kern_sd = int(round(80 / dataset_obj.bin_width))
    window_80ms = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window_80ms /= np.sum(window_80ms)
    filt_80 = lambda x: np.convolve(x, window_80ms, 'same')

    test_ho_30_smth_spikes = np.apply_along_axis(filt_30, 0, test_ho_segments[:, -1, :])
    test_hi_30_smth_spikes = np.apply_along_axis(filt_30, 0, test_hi_segments[:, -1, :])
    test_ho_50_smth_spikes = np.apply_along_axis(filt_50, 0, test_ho_segments[:, -1, :])
    test_hi_50_smth_spikes = np.apply_along_axis(filt_50, 0, test_hi_segments[:, -1, :])
    test_ho_80_smth_spikes = np.apply_along_axis(filt_80, 0, test_ho_segments[:, -1, :])
    test_hi_80_smth_spikes = np.apply_along_axis(filt_80, 0, test_hi_segments[:, -1, :])
    
    train_dict = {
        'train_spikes_heldin': train_hi_segments,
        'train_spikes_heldout': train_ho_segments,
        'train_vel_segments': train_vel_segments
    }

    test_dict = {
        'test_spikes_heldin': test_hi_segments,
        'test_spikes_heldout': test_ho_segments,
        'test_ho_30_smth_spikes': test_ho_30_smth_spikes,
        'test_hi_30_smth_spikes': test_hi_30_smth_spikes,
        'test_ho_50_smth_spikes': test_ho_50_smth_spikes,
        'test_hi_50_smth_spikes': test_hi_50_smth_spikes,
        'test_ho_80_smth_spikes': test_ho_80_smth_spikes,
        'test_hi_80_smth_spikes': test_hi_80_smth_spikes,
        'test_vel_segments': test_vel_segments
    }

    h5_file = {**train_dict, **test_dict}
    filename = f'{data_dir}{dataset}_{seq_len}_{overlap}_{lag}.h5'

    # Remove older version if it exists
    if osp.isfile(filename): os.remove(filename)
    save_to_h5(h5_file, filename, overwrite=True)
