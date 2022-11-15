#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import copy
import random
import logging
import requests
import pandas as pd
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
from sklearn.model_selection import KFold
from nlb_tools.make_tensors import (
    save_to_h5,
    combine_h5,
    h5_to_dict,
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors)
from utils_f import set_seeds
from data.t5_dataset import T5CursorDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler



import warnings; warnings.simplefilter('ignore')
'''────────────────────────────── datasets.py ───────────────────────────────'''



datasets = {}

def populate_datasets(config):
    ''' Create dataset object for each session and store
    '''
    if datasets == {}:
        for session in [*config.data.pretrain_sessions, *config.data.finetune_sessions]:
            if not session in datasets:
                datasets[session] = T5CursorDataset(f'{config.data.dir}/{session}.mat')

def chop(data, seq_len, overlap):
    ''' Chop data function
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

def get_testing_data(config):
    ''' Get spikes to train NDT with
    '''
    # Cache datasets for later use
    populate_datasets(config)

    return datasets

    trials_dict = {}
    mean_dict = {}
    std_dict = {}
    scaler_dict = {}
    heldin_dict = {}
    heldout_dict = {}
    corr_chan_dict = {}

    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')
    trials = {}
    chopped_spikes, session_names = [], []
    for session in config.data.pretrain_sessions:
        scaler_dict[session] = []
        dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

        session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

        corr_chan_dict[session]=[]
        if config.data.rem_xcorr: 
            corr_chan_dict[session] = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)[1]

        dataset.resample(config.data.bin_size / 1000)
        dataset.smooth_spk(20, name='smth')

        failed_trials = ~dataset.trial_info['is_successful'] 
        center_trials = dataset.trial_info['is_center_target']
        ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
        cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()

        spks = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.index
        spks_idx = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes

        n_channels = dataset.data.spikes.shape[-1]

        n_heldout = int(config.data.heldout_pct * n_channels)
        np.random.seed(config.setup.seed)
        heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
        heldin_channels = torch.ones(n_channels, dtype=bool)
        heldin_channels[heldout_channels] = False
        heldin_dict[session] = heldin_channels
        heldout_dict[session] = heldout_channels
        
        mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.mean(0)
        std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.std(0)
        # mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.mean(0)
        # std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.std(0)
        data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.to_numpy()
        # data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.to_numpy()
        
        # for row in range(data.shape[1]):
        #     temp = StandardScaler()
        #     temp.fit(data[:, row].reshape(-1, 1))
        #     scaler_dict[session].append(temp)

        trial_data = dataset.make_trial_data(
            align_field='start_time',
            align_range=(
                0,
                # -config.data.seq_len * config.data.bin_size + config.data.bin_size, # start align
                config.data.trial_len + config.data.lag
                # 0, config.data.trial_len
            ),
            allow_overlap=True,
            ignored_trials=failed_trials | center_trials | cl_blocks
        )

        trial_data.sort_index(axis=1, inplace=True)
        trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
        trial_data['condition'] = 0

        for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id


        trials_dict[session] = {}
        for cond_id, trials in trial_data.groupby('condition'):
            trial_list = []
            for trial_id, trial in trials.groupby('trial_id'):
                trial_list.append(trial.spikes.to_numpy()[:, heldin_channels])
            trials_dict[session][cond_id] = trial_list

    return trials_dict, datasets
    # return trials_dict, mean_dict, std_dict, scaler_dict, heldin_dict, heldout_dict, corr_chan_dict

# def get_testing_data(config):
#     ''' Get spikes to train NDT with
#     '''
#     # Cache datasets for later use
#     populate_datasets(config)

#     trials_dict = {}
#     mean_dict = {}
#     std_dict = {}
#     scaler_dict = {}
#     heldin_dict = {}
#     heldout_dict = {}
#     corr_chan_dict = {}

#     session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')
#     trials = {}
#     chopped_spikes, session_names = [], []
#     for session in config.data.pretrain_sessions:
#         scaler_dict[session] = []
#         dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

#         session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

#         corr_chan_dict[session]=[]
#         if config.data.rem_xcorr: 
#             corr_chan_dict[session] = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)[1]

#         dataset.resample(config.data.bin_size / 1000)
#         dataset.smooth_spk(20, name='smth')

#         failed_trials = ~dataset.trial_info['is_successful'] 
#         center_trials = dataset.trial_info['is_center_target']
#         ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
#         cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()

#         spks = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.index
#         spks_idx = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes

#         n_channels = dataset.data.spikes.shape[-1]

#         n_heldout = int(config.data.heldout_pct * n_channels)
#         np.random.seed(config.setup.seed)
#         heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
#         heldin_channels = torch.ones(n_channels, dtype=bool)
#         heldin_channels[heldout_channels] = False
#         print(heldout_channels)
#         heldin_dict[session] = heldin_channels
#         heldout_dict[session] = heldout_channels
        
#         mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.mean(0)
#         std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.std(0)
#         # mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.mean(0)
#         # std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.std(0)
#         data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.to_numpy()
#         # data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.to_numpy()
        
#         # for row in range(data.shape[1]):
#         #     temp = StandardScaler()
#         #     temp.fit(data[:, row].reshape(-1, 1))
#         #     scaler_dict[session].append(temp)

#         trial_data = dataset.make_trial_data(
#             align_field='start_time',
#             align_range=(
#                 0,
#                 # -config.data.seq_len * config.data.bin_size + config.data.bin_size, # start align
#                 config.data.trial_len + config.data.lag
#                 # 0, config.data.trial_len
#             ),
#             allow_overlap=True,
#             ignored_trials=failed_trials | center_trials | cl_blocks
#         )

#         trial_data.sort_index(axis=1, inplace=True)
#         trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
#         trial_data['condition'] = 0

#         for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):
#             indices = trial_data.index[trial_data['X&Y'] == xy]
#             trial_data.loc[indices, 'condition'] = id


#         trials_dict[session] = {}
#         for cond_id, trials in trial_data.groupby('condition'):
#             trial_list = []
#             for trial_id, trial in trials.groupby('trial_id'):
#                 trial_list.append(trial.spikes.to_numpy()[:, heldin_channels])
#             trials_dict[session][cond_id] = trial_list

#     return trials_dict, mean_dict, std_dict, scaler_dict, heldin_dict, heldout_dict, corr_chan_dict
    
# def get_testing_data(config):
#     ''' Get spikes to train NDT with
#     '''
#     # Cache datasets for later use
#     populate_datasets(config)

#     trials_dict = {}
#     mean_dict = {}
#     std_dict = {}
#     scaler_dict = {}
#     heldin_dict = {}
#     heldout_dict = {}
#     corr_chan_dict = {}

#     session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')
#     trials = {}
#     chopped_spikes, session_names = [], []
#     for session in config.data.pretrain_sessions:
#         scaler_dict[session] = []
#         dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

#         session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

#         corr_chan_dict[session]=[]
#         if config.data.rem_xcorr: 
#             corr_chan_dict[session] = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)[1]

#         dataset.resample(config.data.bin_size / 1000)
#         dataset.smooth_spk(20, name='smth')

#         failed_trials = ~dataset.trial_info['is_successful'] 
#         center_trials = dataset.trial_info['is_center_target']
#         ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
#         cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()

#         n_channels = dataset.data.spikes.shape[-1]

#         n_heldout = int(config.data.heldout_pct * n_channels)
#         np.random.seed(config.setup.seed)
#         heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
#         heldin_channels = torch.ones(n_channels, dtype=bool)
#         heldin_channels[heldout_channels] = False
#         print(heldout_channels)
#         heldin_dict[session] = heldin_channels
#         heldout_dict[session] = heldout_channels
        
#         mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.mean(0)
#         std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.std(0)
#         # mean_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.mean(0)
#         # std_dict[session] = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.std(0)
#         data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.to_numpy()
#         # data = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.to_numpy()
        
#         # for row in range(data.shape[1]):
#         #     temp = StandardScaler()
#         #     temp.fit(data[:, row].reshape(-1, 1))
#         #     scaler_dict[session].append(temp)

#         trial_data = dataset.make_trial_data(
#             align_field='start_time',
#             align_range=(
#                 0,
#                 # -config.data.seq_len * config.data.bin_size + config.data.bin_size, # start align
#                 config.data.trial_len + config.data.lag
#                 # 0, config.data.trial_len
#             ),
#             allow_overlap=True,
#             ignored_trials=failed_trials | center_trials | cl_blocks
#         )

#         trial_data.sort_index(axis=1, inplace=True)
#         trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
#         trial_data['condition'] = 0

#         for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):
#             indices = trial_data.index[trial_data['X&Y'] == xy]
#             trial_data.loc[indices, 'condition'] = id


#         trials_dict[session] = {}
#         for cond_id, trials in trial_data.groupby('condition'):
#             trial_list = []
#             for trial_id, trial in trials.groupby('trial_id'):
#                 trial_list.append(trial.spikes.to_numpy()[:, heldin_channels])
#             trials_dict[session][cond_id] = trial_list

#     return trials_dict, mean_dict, std_dict, scaler_dict, heldin_dict, heldout_dict, corr_chan_dict
    
def get_pretraining_data(config):
    ''' Get spikes to train NDT with
    '''
    # Cache datasets for later use
    populate_datasets(config)

    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

    chopped_spikes, session_names = [], []
    for session in config.data.pretrain_sessions:
        dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

        if config.data.rem_xcorr: 
            pair_corr, chan_names_to_drop = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)

        dataset.resample(config.data.bin_size / 1000) # convert ms to sec

        block_spikes = []
        ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item()
        for block_num, block in dataset.data.groupby(('blockNums', 'n')):
            if block_num == ol_block:
                block_spikes.append(chop(block.spikes.to_numpy(), config.data.seq_len, config.data.overlap))
        spikes = np.concatenate(block_spikes, 0)

        names = [session for i in range(spikes.shape[0])]

        chopped_spikes.append(torch.from_numpy(spikes.astype(np.float32)))
        session_names.append(names)
    return torch.cat(chopped_spikes, 0), np.concatenate(session_names, 0)
    # Should this be removed ^^

    # Add check to see if heldout has the same distribution across datasets
        # at the same time add a caching system for params / datset to save a local copy
        # this involves saving the paramters in the config that affect the dataset and if they have changed,
        #  store another copy

trials_dict = None

def get_trial_data():
    global trials_dict
    return trials_dict

def get_alignment_matricies(config):
    ''' 
    '''
    global trials_dict
    # Cache datasets for later use
    populate_datasets(config)

    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

    avg_conds = {}
    trials_dict = {}
    cond_list = None

    cond_list = None

    for session in config.data.pretrain_sessions:
        dataset = copy.deepcopy(datasets[session])

        if config.data.rem_xcorr: 
            dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)

        dataset.resample(config.data.bin_size / 1000)
        dataset.smooth_spk(config.data.smth_std, name='smth')

        failed_trials = ~dataset.trial_info['is_successful'] 
        center_trials = dataset.trial_info['is_center_target']
        ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
        cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()
        
        trial_data = dataset.make_trial_data(
            align_field='start_time',
            align_range=(0, config.data.trial_len),
            allow_overlap=True,
            ignored_trials=failed_trials | center_trials | cl_blocks
        )

        trial_data.sort_index(axis=1, inplace=True)
        trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
        trial_data['condition'] = 0

        if cond_list == None:
            cond_list = list(zip(trial_data['X&Y'].unique(), np.arange(1,9)))
        for xy, id in cond_list:    
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id

        n_channels = trial_data.spikes.shape[-1]

        n_heldout = int(config.data.heldout_pct * n_channels)
        np.random.seed(config.setup.seed)
        heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
        heldin_channels = torch.ones(n_channels, dtype=bool)
        heldin_channels[heldout_channels] = False

        avg_conds[session] = []
        trials_dict[session] = {}
        for cond_id, trials in trial_data.groupby('condition'):
            trial_list = []
            smth_trial_list = []
            for trial_id, trial in trials.groupby('trial_id'):
                trial_list.append(trial.spikes.to_numpy()[:, heldin_channels])
                # all_ch = np.concatenate((trial.spikes.to_numpy()[:, heldin_channels], np.zeros_like(trial.spikes.to_numpy()[:, heldout_channels])), -1)
                # trial_list.append(all_ch)
                smth_trial_list.append(trial.spikes_smth.to_numpy()[:, heldin_channels])
                # all_ch_smth = np.concatenate((trial.spikes_smth.to_numpy()[:, heldin_channels], np.zeros_like(trial.spikes_smth.to_numpy()[:, heldout_channels])), -1)
                # smth_trial_list.append(all_ch_smth)
            trials_dict[session][cond_id] = trial_list
            avg_conds[session].append(np.mean(smth_trial_list, 0))

    session_list = list(avg_conds.keys())

    if config.model.rand_readin_init:
        return None, None, session_list

    avg_cond_arr = np.array(list(avg_conds.values())) # (days, conds, bins, chans)
    avg_cond_arr = avg_cond_arr.transpose((3, 0, 1, 2)) # -> (chans, days, conds, bins)
    nchans, ndays, nconds, nbins = avg_cond_arr.shape
    avg_cond_arr = avg_cond_arr.reshape((nchans * ndays, nconds * nbins))

    avg_cond_means = avg_cond_arr.mean(axis=1)
    avg_cond_centered = (avg_cond_arr.T - avg_cond_means.T).T

    pca = PCA(n_components=config.model.factor_dim)
    pca.fit(avg_cond_centered.T)

    dim_reduced_data = np.dot(avg_cond_centered.T, pca.components_.T).T
    avg_cond_arr = avg_cond_arr.reshape((nchans, ndays, nconds, nbins))

    dim_reduced_data_means = dim_reduced_data.mean(axis=1)
    dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T).T

    alignment_matrices = []
    alignment_biases = []
    
    for day in range(ndays):
        this_dataset_data = avg_cond_arr.reshape((nchans, ndays, nconds * nbins))[:, day, :].squeeze()

        this_dataset_means = avg_cond_means.reshape(nchans, ndays)[:, day].squeeze()
        this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(this_dataset_centered.T, dim_reduced_data_this.T)

        alignment_matrices.append(torch.from_numpy(np.copy(reg.coef_.astype(np.float32))))  # nchans x nPCs
        bias = -1 * np.dot(this_dataset_means, reg.coef_.T)
        alignment_biases.append(torch.from_numpy(bias.astype(np.float32)))

    return alignment_matrices, alignment_biases, session_list


class Dataset(data.Dataset):
    def __init__(self, config):
        '''init Dataset

        Args:
            config (dict): A config object.
            filename (str): path to the dataset.h5 file.
            mode (str): ['test', 'train' , 'val', 'trainval'] The dataset type.
        '''
        super().__init__()
        device = torch.device('cuda:0')

        chopped_spikes, session_names = get_pretraining_data(config)

        self.names = session_names
        self.n_samples = chopped_spikes.shape[0]
        self.n_channels = chopped_spikes.shape[-1]

        self.has_heldout = config.data.heldout_pct > 0.0
        
        # If the percentage of heldout channels is above 0, hold out some channels
        if self.has_heldout: 
            n_heldout = int(config.data.heldout_pct * self.n_channels)

            np.random.seed(config.setup.seed)
            heldout_channels = np.random.choice(self.n_channels, n_heldout, replace=False)
            heldin_channels = torch.ones(self.n_channels, dtype=bool)
            heldin_channels[heldout_channels] = False

            self.heldin_spikes = chopped_spikes[:, :, heldin_channels].to(device)
            self.heldout_spikes = chopped_spikes[:, :, heldout_channels].to(device)

            self.n_heldin = self.n_channels - n_heldout
            self.n_heldout = n_heldout
            self.max_train_spks = self.heldin_spikes.max().item() + 3
        else:
            self.spikes = chopped_spikes.to(device)
            self.max_train_spks = self.spikes.max().item() + 3

    def __len__(self):
        '''
        Returns:
            int: The number of samples or trials.
        '''
        return self.n_samples

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index into the batches.
        Returns:
            spikes_heldin (np.ndarray): Spikes held-in.
            spikes_heldout (np.ndarray): Spikes held-out.
            spikes_all_fp (np.ndarray): Spikes forward pass.
        '''
        if self.has_heldout: 
            return self.heldin_spikes[index], self.heldout_spikes[index], self.names[index]
        else:
            return self.spikes[index], self.names[index]

    def get_dataloader(self, generator, shuffle=True):
        return data.DataLoader(
            self,
            batch_size=self.config['train']['batch_size'],
            generator=generator,
            shuffle=shuffle
        )

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
    generator = torch.Generator()
    generator.manual_seed(config['setup']['seed'])

    def _init_fn(worker_id):
        set_seeds(config)
    
    if mode == 'train_val':
        train_data = Dataset(config)
        split_index = int(0.8 * train_data.n_samples)

        if config['train']['val_type'] == 'last':
            # Dataset is last 20% of training data
            train_indicies = [i for i in range(0, split_index)]
            val_indicies = [i for i in range(split_index, train_data.n_samples)]

        if config['train']['val_type'] == 'random':
            shuffled_indicies = torch.randperm(train_data.n_samples, generator=generator)
            train_indicies = list(shuffled_indicies[:split_index])
            val_indicies = list(shuffled_indicies[split_index:])

        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, train_indicies),
            batch_size=config['train']['batch_size'],
            generator=generator,
            worker_init_fn=_init_fn,
            shuffle=True,
        )

        if train_data.has_heldout:
            val_data = (
                train_data.heldin_spikes[val_indicies], 
                train_data.heldout_spikes[val_indicies],
                train_data.names[val_indicies]
            ) 
        else:
            val_data = train_data.spikes[val_indicies], train_data.names[val_indicies]

        return train_dataloader, val_data
    
    # KFOLDS should also have a val set
    elif mode == 'cross_val':
        train_data = Dataset(config)

        train_dataloader = train_data.get_dataloader(generator, shuffle=True)
        fold_dataloaders = []

        kf = KFold(
            n_splits=config['train']['n_folds'], 
            random_state=config['setup']['seed'], 
            shuffle=True
        )

        for idx, (train_indicies, val_indicies) in enumerate(kf.split(train_data)):
            tmp_train_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data, train_indicies),
                batch_size=config['train']['batch_size'],
                generator=generator,
                worker_init_fn=_init_fn,
                shuffle=True
            )

            tmp_val_dataloader = (train_data.spikes_heldin[val_indicies], train_data.spikes_heldout[val_indicies])

            fold_dataloaders.append((idx, tmp_train_dataloader, tmp_val_dataloader))

        return train_dataloader, fold_dataloaders

#  OLD

# def get_dataloaders(config, mode):
#     ''' Creates Dataset objects and creates the DataLoaders from them.

#     Args:
#         config (dict): A config object.
#         mode (str): ['original', 'random', 'test', 'none'] Which validation set
#                     shold be used. Original is the nlb given validation set,
#                     random is a random subset of the combined train and
#                     validation set, none is no validation set
#     Returns:
#         train_dataloader (DataLoader): The training set DataLoader.
#         (test/val)_dataloader (DataLoader, None): The test or validation set
#                                                   DataLoader. In 'none' this is
#                                                   None.
#     '''
#     data_path = '{0}{1}_{2}_{3}_{4}.h5'.format(
#         config["setup"]["data_dir"], 
#         config["setup"]["dataset"], 
#         config["train"]["seq_len"], 
#         config["train"]["overlap"], 
#         config["train"]["lag"]
#     )

#     generator = torch.Generator()
#     generator.manual_seed(config['setup']['seed'])

#     def _init_fn(worker_id):
#         set_seeds(config)
    
#     if mode == 'train_val':
#         train_data = Dataset(config, data_path, 'train')

#         n_samples = len(train_data)
#         split_index = int(0.8 * n_samples)

#         if config['train']['val_type'] == 'last':
#             # Dataset is last 20% of training data
#             train_indicies = [i for i in range(0, split_index)]
#             val_indicies = [i for i in range(split_index, n_samples)]

#         if config['train']['val_type'] == 'random':
#             shuffled_indicies = torch.randperm(n_samples, generator=generator)
#             train_indicies = list(shuffled_indicies[:split_index])
#             val_indicies = list(shuffled_indicies[split_index:])

#         train_dataloader = torch.utils.data.DataLoader(
#             torch.utils.data.Subset(train_data, train_indicies),
#             batch_size=config['train']['batch_size'],
#             generator=generator,
#             worker_init_fn=_init_fn,
#             shuffle=True,
#         )

#         val_dataloader = (train_data.spikes_heldin[val_indicies], train_data.spikes_heldout[val_indicies])

#         return train_dataloader, val_dataloader
    
#     elif mode == 'cross_val':
#         train_data = Dataset(config, data_path, 'train')

#         train_dataloader = train_data.get_dataloader(generator, shuffle=True)
#         fold_dataloaders = []

#         kf = KFold(
#             n_splits=config['train']['n_folds'], 
#             random_state=config['setup']['seed'], 
#             shuffle=True
#         )

#         for idx, (train_indicies, val_indicies) in enumerate(kf.split(train_data)):
#             tmp_train_dataloader = torch.utils.data.DataLoader(
#                 torch.utils.data.Subset(train_data, train_indicies),
#                 batch_size=config['train']['batch_size'],
#                 generator=generator,
#                 worker_init_fn=_init_fn,
#                 shuffle=True
#             )

#             tmp_val_dataloader = (train_data.spikes_heldin[val_indicies], train_data.spikes_heldout[val_indicies])

#             fold_dataloaders.append((idx, tmp_train_dataloader, tmp_val_dataloader))

#         return train_dataloader, fold_dataloaders


def chop_data(data, chopsize, overlap, lag_bins, single_trail=False):
    ''' Chops data trial by trail (or segment by segment) into overlapping segments.'''
    
    def chop_trial(trial):
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
        return np.lib.stride_tricks.as_strided(trial, shape=shape, strides=strides).copy().astype('f')
    
    if single_trail:
        return chop_trial(np.array(data))
    else:
        chopped_data = []

        for trial in data:
            chopped_data.append(chop_trial(trial))

        chopped_data = np.array(chopped_data)
        return chopped_data.reshape((
            chopped_data.shape[0] * chopped_data.shape[1], 
            chopped_data.shape[2], 
            chopped_data.shape[3]
        ))


# class Dataset(data.Dataset):
#     def __init__(self, config, filename, mode):
#         '''init Dataset

#         Args:
#             config (dict): A config object.
#             filename (str): path to the dataset.h5 file.
#             mode (str): ['test', 'train' , 'val', 'trainval'] The dataset type.
#         '''
#         super().__init__()
        
#         with h5py.File(filename, 'r') as h5file:
#             h5dict = h5_to_dict(h5file)
#             self.config = config
#             self.mode = mode

#             device = torch.device('cuda:0')

#             def set_sizes(self):
#                 ''' Helper function that assigns the number of samples, trial
#                 length, forward pass length, number of heldin neurons, and
#                 number of heldout neurons.
#                 '''
#                 self.n_samples = self.spikes_heldin.shape[0]
#                 self.n_heldin = self.spikes_heldin.shape[2]
#                 self.n_heldout = self.spikes_heldout.shape[2]
#                 self.n_neurons = self.n_heldin + self.n_heldout
                
#                 self.max_train_spks = h5dict['train_spikes_heldin'].max().item() + 3

#             if mode == 'train':
#                 self.spikes_heldin = torch.Tensor(h5dict['train_spikes_heldin'].astype(np.float32)).to(device)
#                 self.spikes_heldout = torch.Tensor(h5dict['train_spikes_heldout'].astype(np.float32)).to(device)

#                 set_sizes(self)

#             elif mode == 'test':
#                 self.spikes_heldin = torch.Tensor(h5dict['test_spikes_heldin'].astype(np.float32)).to(device)
#                 self.spikes_heldout = torch.Tensor(h5dict['test_spikes_heldout'].astype(np.float32)).to(device)

#                 set_sizes(self)

#     def __len__(self):
#         '''
#         Returns:
#             int: The number of samples or trials.
#         '''
#         return self.spikes_heldin.shape[0]

#     def __getitem__(self, index):
#         '''
#         Args:
#             index (int): Index into the batches.
#         Returns:
#             spikes_heldin (np.ndarray): Spikes held-in.
#             spikes_heldout (np.ndarray): Spikes held-out.
#             spikes_all_fp (np.ndarray): Spikes forward pass.
#         '''
#         return (
#             self.spikes_heldin[index],
#             self.spikes_heldout[index]
#         )

#     def clip_max(self, max, indicies):
#         self.spikes_heldin[indicies] = torch.clamp(self.spikes_heldin[indicies], max=max)

#     def get_dataloader(self, generator, shuffle=True):
#         return data.DataLoader(self,
#             batch_size=self.config['train']['batch_size'],
#             generator=generator,
#             shuffle=shuffle)

# def verify_dataset(config):
#     '''Checks if the correct dataset.h5 file is downloaded, if it isn't then it
#     prompts the user if they would like to download it.

#     Args:
#         config (dict): A config object.
#     '''
#     path = f'{config["setup"]["data_dir"]}{config["setup"]["dataset"]}'
#     path += f'_{config["train"]["seq_len"]}_{config["train"]["overlap"]}'
#     path += f'_{config["train"]["lag"]}.h5'

#     if os.path.isfile(path):
#         return None
#     else:
#         print(f'Dataset could not be found at: {path}')
#         response = input('Would you like to create it? (y/n): ')
#         while response != 'y' and response != 'n':
#             response = input("Please enter 'y' or 'n': ")
#         if response == 'n': exit()

#         if config['setup']['dataset'] == 'mc_rtt':
#             download_mc_rtt(config['setup']['data_dir'])

#         create_h5_data(
#             config["setup"]["data_dir"], 
#             config["setup"]["dataset"],
#             config["train"]["seq_len"],
#             config["train"]["overlap"],
#             config["train"]["lag"]
#         )

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


def create_h5_data(data_dir, dataset_name, seq_len, overlap, lag):
    ''' Turn train/test '.nwb' files into a single .h5 file and save
    '''
    filepath = f'{data_dir}{dataset_name}.nwb'

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

    train_hi_segments = chop_data(hi_spike_segments[:3], seq_len, overlap, lag_bins) # (8097, 30, 98)
    train_ho_segments = chop_data(ho_spike_segments[:3], seq_len, overlap, lag_bins) # (8097, 30, 32)
    all_train_hi_segments = chop_data(hi_spike_segments[:3], seq_len, seq_len - 1, lag_bins) # (8097, 30, 98)
    all_train_ho_segments = chop_data(ho_spike_segments[:3], seq_len, seq_len - 1, lag_bins) # (8097, 30, 32)

    test_hi_segments = chop_data(np.expand_dims(hi_spike_segments[3], 0), seq_len, seq_len - 1, lag_bins) # (16191, 30, 98)
    test_ho_segments = chop_data(np.expand_dims(ho_spike_segments[3], 0), seq_len, seq_len - 1, lag_bins) # (16191, 30, 32)

    train_vel = chop_data(train_vel_segments, seq_len, seq_len - 1, lag_bins=0)[:, -1, :]
    test_vel = chop_data(test_vel_segments, seq_len, seq_len - 1, lag_bins=0)[:, -1, :]

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

    dataset_obj = NWBDataset(filepath)

    has_change = dataset_obj.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
    change_nan = dataset_obj.data[has_change].isna().any(axis=1)
    change_times = dataset_obj.data.index[has_change]
    mask = (change_nan.index >= "00:08:06.900000")

    def format_data(dataset, drop_mask):
        dataset_new = copy.deepcopy(dataset)

        start_times = change_times[:-1][~drop_mask]
        end_times = change_times[1:][~drop_mask]
        target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
        reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
        reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180

        dataset_new.trial_info = pd.DataFrame({
            'trial_id': np.arange(len(start_times)),
            'start_time': start_times,
            'end_time': end_times,
            'target_pos': target_pos,
            'reach_dist_x': reach_dist[:, 0],
            'reach_dist_y': reach_dist[:, 1],
            'reach_angle': reach_angle,
        })

        dataset_new.resample(10)

        speed = np.linalg.norm(dataset_new.data.finger_vel, axis=1)

        dataset_new.data['speed_le'] = speed
        peak_times = dataset_new.calculate_onset(
            'speed_le', 
            onset_threshold=0.075, 
            peak_prominence=0.9,
            peak_distance_s=0.9,
            multipeak_threshold=0.9
        )

        dataset_new.data['speed_me'] = speed
        peak_times = dataset_new.calculate_onset(
            'speed_me', 
            onset_threshold=0.075, 
            peak_prominence=0.5,
            peak_distance_s=0.125,
            multipeak_threshold=0.5
        )

        dataset_new.data['speed_he'] = speed
        peak_times = dataset_new.calculate_onset(
            'speed_he', 
            onset_threshold=0.075, 
            peak_prominence=0.1,
            peak_distance_s=0.1,
            multipeak_threshold=0.2
        )

        return dataset_new

    drop_mask_train = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True) | mask)[:-1]
    drop_mask_test = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True) | ~mask)[:-1]

    dataset_train = format_data(dataset_obj, drop_mask_train)
    dataset_test = format_data(dataset_obj, drop_mask_test)

    def run_trial(trial_data, dataset):
        trial_hi, trial_ho, trial_ids, trial_vel, trial_idx = [], [], [], [], []
        for tid, trial in trial_data.groupby('trial_id'):
            trial_hi.append(chop_data(trial.spikes, seq_len, seq_len - 1, lag_bins, True))
            trial_ho.append(chop_data(trial.heldout_spikes, seq_len, seq_len - 1, lag_bins, True))
            trial_ids.append(tid)
            trial_vel.append(trial.finger_vel.to_numpy()[lag_bins + seq_len - 1:])
        idx = 0
        for trial in trial_hi[:-1]:
            idx += trial.shape[0]
            trial_idx.append(idx)

        trial_angles = []
        for tid in np.unique(trial_ids):
            trial_angles.append(dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle.item())
        
        return trial_hi, trial_ho, trial_ids, trial_vel, trial_idx, trial_angles

    def run_trials(train_trial_data, test_trial_data):
        train_res = run_trial(train_trial_data, dataset_train)
        test_res = run_trial(test_trial_data, dataset_test)

        return train_res, test_res

    time_chop_off = (seq_len * -10) + 10

    # All Trials
    all_train_trial_data = dataset_train.make_trial_data(align_range=(time_chop_off, lag), allow_nans=False, allow_overlap=True)
    all_test_trial_data = dataset_test.make_trial_data(align_range=(time_chop_off, lag), allow_nans=False, allow_overlap=True)
    
    all_train_res, all_test_res = run_trials(all_train_trial_data, all_test_trial_data)

    all_train_trial_hi = all_train_res[0]
    all_train_trial_ho = all_train_res[1]
    all_train_trial_ids = all_train_res[2]
    all_train_trial_vel = all_train_res[3]
    all_train_trial_idx = all_train_res[4]
    all_train_trial_angles = all_train_res[5]

    all_test_trial_hi = all_test_res[0]
    all_test_trial_ho = all_test_res[1]
    all_test_trial_ids = all_test_res[2]
    all_test_trial_vel = all_test_res[3]
    all_test_trial_idx = all_test_res[4]
    all_test_trial_angles = all_test_res[5]

    # LE Trials
    le_train_trial_data = dataset_train.make_trial_data(align_field='speed_le_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
    le_test_trial_data = dataset_test.make_trial_data(align_field='speed_le_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
    
    le_train_res, le_test_res = run_trials(le_train_trial_data, le_test_trial_data)

    le_train_trial_hi = le_train_res[0]
    le_train_trial_ho = le_train_res[1]
    le_train_trial_ids = le_train_res[2]
    le_train_trial_vel = le_train_res[3]
    le_train_trial_idx = le_train_res[4]
    le_train_trial_angles = le_train_res[5]

    le_test_trial_hi = le_test_res[0]
    le_test_trial_ho = le_test_res[1]
    le_test_trial_ids = le_test_res[2]
    le_test_trial_vel = le_test_res[3]
    le_test_trial_idx = le_test_res[4]
    le_test_trial_angles = le_test_res[5]

    # ME Trials
    me_train_trial_data = dataset_train.make_trial_data(align_field='speed_me_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
    me_test_trial_data = dataset_test.make_trial_data(align_field='speed_me_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
    
    me_train_res, me_test_res = run_trials(me_train_trial_data, me_test_trial_data)

    me_train_trial_hi = me_train_res[0]
    me_train_trial_ho = me_train_res[1]
    me_train_trial_ids = me_train_res[2]
    me_train_trial_vel = me_train_res[3]
    me_train_trial_idx = me_train_res[4]
    me_train_trial_angles = me_train_res[5]

    me_test_trial_hi = me_test_res[0]
    me_test_trial_ho = me_test_res[1]
    me_test_trial_ids = me_test_res[2]
    me_test_trial_vel = me_test_res[3]
    me_test_trial_idx = me_test_res[4]
    me_test_trial_angles = me_test_res[5]

    # HE Trials
    he_train_trial_data = dataset_train.make_trial_data(align_field='speed_he_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
    he_test_trial_data = dataset_test.make_trial_data(align_field='speed_he_onset', align_range=(time_chop_off, 420 + lag), allow_nans=False, allow_overlap=True)
        
    he_train_res, he_test_res = run_trials(he_train_trial_data, he_test_trial_data)

    he_train_trial_hi = he_train_res[0]
    he_train_trial_ho = he_train_res[1]
    he_train_trial_ids = he_train_res[2]
    he_train_trial_vel = he_train_res[3]
    he_train_trial_idx = he_train_res[4]
    he_train_trial_angles = he_train_res[5]

    he_test_trial_hi = he_test_res[0]
    he_test_trial_ho = he_test_res[1]
    he_test_trial_ids = he_test_res[2]
    he_test_trial_vel = he_test_res[3]
    he_test_trial_idx = he_test_res[4]
    he_test_trial_angles = he_test_res[5]

    train_dict = {
        'train_spikes_heldin': train_hi_segments,
        'train_spikes_heldout': train_ho_segments,
        'all_train_spikes_heldin': all_train_hi_segments,
        'all_train_spikes_heldout': all_train_ho_segments,
        'train_vel': train_vel,

        'all_train_trial_hi': np.concatenate(all_train_trial_hi, axis=0),
        'all_train_trial_ho': np.concatenate(all_train_trial_ho, axis=0),
        'all_train_trial_vel': np.concatenate(all_train_trial_vel, axis=0),
        'all_train_trial_idx': np.array(all_train_trial_idx),
        'all_train_trial_angles': np.array(all_train_trial_angles),

        'le_train_trial_hi': np.concatenate(le_train_trial_hi, axis=0),
        'le_train_trial_ho': np.concatenate(le_train_trial_ho, axis=0),
        'le_train_trial_vel': np.concatenate(le_train_trial_vel, axis=0),
        'le_train_trial_idx': np.array(le_train_trial_idx),
        'le_train_trial_angles': np.array(le_train_trial_angles),

        'me_train_trial_hi': np.concatenate(me_train_trial_hi, axis=0),
        'me_train_trial_ho': np.concatenate(me_train_trial_ho, axis=0),
        'me_train_trial_vel': np.concatenate(me_train_trial_vel, axis=0),
        'me_train_trial_idx': np.array(me_train_trial_idx),
        'me_train_trial_angles': np.array(me_train_trial_angles),

        'he_train_trial_hi': np.concatenate(he_train_trial_hi, axis=0),
        'he_train_trial_ho': np.concatenate(he_train_trial_ho, axis=0),
        'he_train_trial_vel': np.concatenate(he_train_trial_vel, axis=0),
        'he_train_trial_idx': np.array(he_train_trial_idx),
        'he_train_trial_angles': np.array(he_train_trial_angles),
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
        'test_vel': test_vel,

        'all_test_trial_hi': np.concatenate(all_test_trial_hi, axis=0),
        'all_test_trial_ho': np.concatenate(all_test_trial_ho, axis=0),        
        'all_test_trial_vel': np.concatenate(all_test_trial_vel, axis=0),
        'all_test_trial_idx': np.array(all_test_trial_idx),
        'all_test_trial_angles': np.array(all_test_trial_angles), 

        'le_test_trial_hi': np.concatenate(le_test_trial_hi, axis=0),
        'le_test_trial_ho': np.concatenate(le_test_trial_ho, axis=0),        
        'le_test_trial_vel': np.concatenate(le_test_trial_vel, axis=0),
        'le_test_trial_idx': np.array(le_test_trial_idx),
        'le_test_trial_angles': np.array(le_test_trial_angles), 

        'me_test_trial_hi': np.concatenate(me_test_trial_hi, axis=0),
        'me_test_trial_ho': np.concatenate(me_test_trial_ho, axis=0),        
        'me_test_trial_vel': np.concatenate(me_test_trial_vel, axis=0),
        'me_test_trial_idx': np.array(me_test_trial_idx),
        'me_test_trial_angles': np.array(me_test_trial_angles), 

        'he_test_trial_hi': np.concatenate(he_test_trial_hi, axis=0),
        'he_test_trial_ho': np.concatenate(he_test_trial_ho, axis=0),        
        'he_test_trial_vel': np.concatenate(he_test_trial_vel, axis=0),
        'he_test_trial_idx': np.array(he_test_trial_idx),
        'he_test_trial_angles': np.array(he_test_trial_angles), 
    }

    h5_file = {**train_dict, **test_dict}
    filename = f'{data_dir}{dataset_name}_{seq_len}_{overlap}_{lag}.h5'

    # Remove older version if it exists
    if osp.isfile(filename): os.remove(filename)
    save_to_h5(h5_file, filename, overwrite=True)
