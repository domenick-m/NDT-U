#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import copy
import pandas as pd
import os.path as osp
#────#
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from utils.t5_utils import load_toolkit_datasets
import pickle as pkl
import h5py

def align_new_session(config):
    trained_mdl_dir = osp.dirname(config.dirs.trained_mdl_path)
    # with open(f'{trained_mdl_dir}/dim_reduced_data.pkl', 'rb') as dfile:
    #     dim_reduced_data = pkl.load(dfile)
    
    # alignment_matrices = []
    # alignment_biases = []

    # avg_cond_arr = np.array(list(avg_conds.values())) # (sessions, conds, bins, chans)
    # avg_cond_arr = avg_cond_arr.transpose((3, 0, 1, 2)) # -> (chans, sessions, conds, bins)
    # nchans, n_sessions, nconds, nbins = avg_cond_arr.shape
    # avg_cond_arr = avg_cond_arr.reshape((nchans * n_sessions, nconds * nbins))

    # avg_cond_means = avg_cond_arr.mean(axis=1)
    # avg_cond_centered = (avg_cond_arr.T - avg_cond_means.T).T
    
    # for session in config.data.sessions:
    #     this_dataset_data = avg_cond_arr.reshape((nchans, n_sessions, nconds * nbins))[:, session, :].squeeze()

    #     this_dataset_means = avg_cond_means.reshape(nchans, n_sessions)[:, session].squeeze()
    #     this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

    #     reg = Ridge(alpha=1.0, fit_intercept=False)
    #     reg.fit(this_dataset_centered.T, dim_reduced_data_this)

    #     alignment_matrices.append(torch.from_numpy(np.copy(reg.coef_.astype(np.float32))))  # n_chans x n_PCs
    #     bias = -1 * np.dot(this_dataset_means, reg.coef_.T)
    #     alignment_biases.append(torch.from_numpy(bias.astype(np.float32)))
    
    # session_list = list(avg_conds.keys())

    # return alignment_matrices, alignment_biases, session_list

def get_alignment_matricies(config):
    ''' 
    '''
    if config.model.rand_readin_init:
        print('\nUsing Random Alignment Matricies...\n')
        return None, None

    datasets = load_toolkit_datasets(config)
    session_csv = pd.read_csv(config.dirs.sess_csv_path)

    alignment_matrices = []
    alignment_biases = []

    if osp.exists(osp.join(config.dirs.save_dir, 'pcr_alignment.h5')):
        print('\nLoading in Alignment Matricies...\n')

        with h5py.File(osp.join(config.dirs.save_dir, 'pcr_alignment.h5'), 'r') as h5:
            for session in config.data.sessions:
                alignment_matrices.append(torch.Tensor(h5[session]['matrix']))
                alignment_biases.append(torch.Tensor(h5[session]['bias']))
            
        return alignment_matrices, alignment_biases
        
    print('\nCreating Alignment Matricies...\n')

    avg_conds = {}
    cond_list = None

    for session in config.data.sessions:
        # get one session
        dataset = datasets[session]

        if config.data.rem_xcorr: 
            dataset.get_pair_xcorr('spikes', threshold=config.data.xcorr_thesh, zero_chans=True)

        dataset.resample(config.data.bin_size / 1000)
        dataset.smooth_spk(config.data.smth_std, name='smth')

        failed_trials = ~dataset.trial_info['is_successful'] 
        center_trials = dataset.trial_info['is_center_target']
        ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item()
        cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()
        
        # trialize data
        trial_data = dataset.make_trial_data(
            align_field='start_time',
            align_range=(0, config.data.trial_len),
            allow_overlap=True,
            ignored_trials=failed_trials | center_trials | cl_blocks
        )

        # combine x and y of target positions
        trial_data.sort_index(axis=1, inplace=True)
        trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
        trial_data['condition'] = 0

        # use same condition list for all sessions
        cond_list = list(zip(trial_data['X&Y'].unique(), np.arange(1,9))) if cond_list is None else cond_list
        for xy, id in cond_list:    
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id

        # get indicies to remove heldout channels
        n_channels = trial_data.spikes.shape[-1]
        n_heldout = int(config.data.pct_heldout * n_channels)
        np.random.seed(config.data.heldout_seed)
        heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
        heldin_channels = torch.ones(n_channels, dtype=bool)
        heldin_channels[heldout_channels] = False

        # condition average data
        avg_conds[session] = []
        for cond_id, trials in trial_data.groupby('condition'):
            smth_trial_list = []
            for trial_id, trial in trials.groupby('trial_id'):
                smth_trial_list.append(trial.spikes_smth.to_numpy()[:, heldin_channels])

            # take the mean of all trials in condition
            cond_avgd_trials = np.mean(smth_trial_list, 0)
            avg_conds[session].append(cond_avgd_trials)

    # turn dataframe into array and reshape
    avg_cond_arr = np.array(list(avg_conds.values())) # (sessions, conds, bins, chans)
    avg_cond_arr = avg_cond_arr.transpose((3, 0, 1, 2)) # -> (chans, sessions, conds, bins)
    nchans, n_sessions, nconds, nbins = avg_cond_arr.shape
    avg_cond_arr = avg_cond_arr.reshape((nchans * n_sessions, nconds * nbins))

    # mean subtract data
    avg_cond_means = avg_cond_arr.mean(axis=1)
    avg_cond_centered = (avg_cond_arr.T - avg_cond_means.T).T

    # run pca to reduce to factor_dim dimensions
    pca = PCA(n_components=config.model.factor_dim)
    pca.fit(avg_cond_centered.T)

    # get reduced dimensonality data
    dim_reduced_data = np.dot(avg_cond_centered.T, pca.components_.T).T
    avg_cond_arr = avg_cond_arr.reshape((nchans, n_sessions, nconds, nbins))

    # mean subtract data
    dim_reduced_data_means = dim_reduced_data.mean(axis=1)
    dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T)

    with h5py.File(osp.join(config.dirs.save_dir, 'pcr_alignment.h5'), 'w') as h5:
        h5.create_dataset('dim_reduced_data', data=dim_reduced_data_this) # nchans x nPCs
        
        # loop through sessions and regress each day to the factors (dim reduced condition averaged data)
        for idx, session in enumerate(config.data.sessions):
            # get one session
            this_dataset_data = avg_cond_arr.reshape((nchans, n_sessions, nconds * nbins))[:, idx, :].squeeze()

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