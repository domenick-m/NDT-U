#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import copy
import pandas as pd
import os.path as osp
#────#
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from utils.toolkit_utils import load_toolkit_datasets, get_trialized_data
import pickle as pkl
import h5py


def align_sessions(config):
    '''
    '''
    alignment_matrices, alignment_biases = [], []

    print('\nCreating Alignment Matricies...\n')
    
    # load each snel_toolkit dataset into dict
    datasets = load_toolkit_datasets(config)

    # make trialized open- and closed-loop data
    trialized_data = get_trialized_data(config, datasets)

    # make sure that each trial is this long
    trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size
    
    # ! TODO ! Make sure above code works with 20ms bin size on trial lengths other than 2000

    cond_avg_data = {}
    for session in config.data.sessions:
        # get one session
        dataset = datasets[session]
        trialized_dataset = trialized_data[session]['ol_trial_data']
        cond_avg_data[session] = []
        
        for cond_id, trials in trialized_dataset.groupby('condition'):
            smth_trial_list = []
            if cond_id != 0:
                for trial_id, trial in trials.groupby('trial_id'):
                    if trial.spikes_smth.shape[0] == trial_len:
                        smth_trial_list.append(trial.spikes_smth.to_numpy()[:, dataset.heldin_channels])

                # take the mean of all trials in condition
                smth_trial_list = np.array(smth_trial_list)
                # print(smth_trial_list.shape)
                cond_avg_trials = np.mean(smth_trial_list, 0)
                cond_avg_data[session].append(cond_avg_trials)

    # turn dataframe into array and reshape
    cond_avg_arr = np.array(list(cond_avg_data.values())) # (sessions, conds, bins, chans)
    cond_avg_arr = cond_avg_arr.transpose((3, 0, 1, 2)) # -> (chans, sessions, conds, bins)
    nchans, n_sessions, nconds, nbins = cond_avg_arr.shape
    cond_avg_arr = cond_avg_arr.reshape((nchans * n_sessions, nconds * nbins))

    # mean subtract data
    avg_cond_means = cond_avg_arr.mean(axis=1)
    avg_cond_centered = (cond_avg_arr.T - avg_cond_means.T).T

    # run pca to reduce to factor_dim dimensions
    pca = PCA(n_components=config.model.factor_dim)
    pca.fit(avg_cond_centered.T)

    # get reduced dimensonality data
    dim_reduced_data = np.dot(avg_cond_centered.T, pca.components_.T).T
    cond_avg_arr = cond_avg_arr.reshape((nchans, n_sessions, nconds, nbins))

    # mean subtract data
    dim_reduced_data_means = dim_reduced_data.mean(axis=1)
    dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T)

    cached_pcr_dir = osp.join(config.dirs.dataset_dir, 'cached_pcr')
    os.makedirs(cached_pcr_dir, exist_ok=True)

    h5_filename = get_alignment_filename(config)

    cached_pcr_path = osp.join(cached_pcr_dir, h5_filename)

    with h5py.File(cached_pcr_path, 'w') as h5:
        h5.create_dataset('dim_reduced_data', data=dim_reduced_data_this) # n_chans x n_PCs
    
        # loop through sessions and regress each day to the factors (dim reduced condition averaged data)
        for idx, session in enumerate(config.data.sessions):
            # get one session
            this_dataset_data = cond_avg_arr.reshape((nchans, n_sessions, nconds * nbins))[:, idx, :].squeeze()

            # mean subtract
            this_dataset_means = avg_cond_means.reshape(nchans, n_sessions)[:, idx].squeeze()
            this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

            # run Ridge regression to fit this session to dim reduced data
            reg = Ridge(alpha=1.0, fit_intercept=False)
            reg.fit(this_dataset_centered.T, dim_reduced_data_this)

            # use the coefficients as the alignment matrix
            matrix = torch.from_numpy(np.copy(reg.coef_.astype(np.float32)))
            alignment_matrices.append(matrix)  # n_chans x n_PCs

            # mean subtract the data after the readin using the bias
            bias = torch.from_numpy((-1 * np.dot(this_dataset_means, reg.coef_.T)).astype(np.float32))
            alignment_biases.append(bias) # n_chans

            group = h5.create_group(session)
            group.create_dataset('matrix', data=matrix)
            group.create_dataset('bias', data=bias)

    return alignment_matrices, alignment_biases


def align_new_session(config):
    '''
    '''
    path = osp.join(osp.dirname(config.dirs.trained_mdl_path), 'config.yaml')
    # this is ...


def load_alignment_matricies(config, path):
    '''
    '''
    alignment_matrices, alignment_biases = [], []
    with h5py.File(path, 'r') as h5:
            for session in config.data.sessions:
                alignment_matrices.append(torch.Tensor(np.array(h5[session]['matrix'])))
                alignment_biases.append(torch.Tensor(np.array(h5[session]['bias'])))
    return alignment_matrices, alignment_biases

def get_alignment_matricies(config):
    ''' 
    '''
    # readins will be randomly initialized
    if config.model.rand_readin_init:
        return None, None

    # if initializing a model to fine tune then load in reduced_dim data to align to
    if config.dirs.trained_mdl_path != '':
        return align_new_session(config)

    # get directory where cached .h5 files are
    cached_pcr_dir = osp.join(config.dirs.dataset_dir, 'cached_pcr')
    os.makedirs(cached_pcr_dir, exist_ok=True)

    # get filename of cached pcr with same parameters
    h5_filename = get_alignment_filename(config)

    # combine directory and filename
    cached_pcr_path = osp.join(cached_pcr_dir, h5_filename)

    # if cached version of pcr exists, then load it in
    if osp.exists(cached_pcr_path):
        print('\nCached Alignment Matricies Found, Loading in...\n')
        return load_alignment_matricies(config, cached_pcr_path)
    
    # create cached alignment matrices and return them
    return align_sessions(config)


def get_alignment_filename(config):

    data = config.data
    model = config.model

    param_list = [
        model.factor_dim, 
        data.smth_std, 
        data.ol_align_field, 
        data.ol_align_range[0], 
        data.ol_align_range[1]
    ]
    param_list += data.sessions

    param_string = ''.join(f'_{param}' for param in param_list)

    h5_filename = f'pcr_{hash(param_string)}.h5'

    return  h5_filename