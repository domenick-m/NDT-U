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
from utils.toolkit_utils import load_toolkit_datasets, get_trialized_data, get_data_filename, get_toolkit_dataset
import pickle as pkl
import h5py
import hashlib




def align_sessions(config):
    '''
    '''
    alignment_matrices, alignment_biases = {}, {}
    
    # load each snel_toolkit dataset into dict
    datasets = load_toolkit_datasets(config)

    # make trialized open- and closed-loop data
    trialized_data = get_trialized_data(config, datasets)

    # make sure that each trial is this long
    trial_type_range = config.data.ol_align_range if config.model.readin_init == 'ol' else config.data.cl_align_range
    trial_len = (trial_type_range[1] - trial_type_range[0]) / config.data.bin_size

    # ! TODO ! Make sure above code works with 20ms bin size on trial lengths other than 2000
    cond_avg_data = {}
    for session in config.data.sessions:
        # get one session
        dataset = copy.deepcopy(datasets[session])
        # dataset = datasets[session]

        trial_type = 'ol_trial_data' if config.model.readin_init == 'ol' else 'cl_trial_data'
        trialized_dataset = trialized_data[session][trial_type]

        cond_avg_data[session] = []
        # for cond_id_, trials in trialized_dataset.groupby(('cond_id', 'n')):
        for cond_id in range(1,9):
            smth_trial_list = []
            for trial_id, trial in trialized_dataset.groupby('trial_id'):
                if datasets[session].trial_info.loc[trial_id].cond_id == cond_id:
                # get position where condition is above 0, filter out -1,-1 targets
                    smth_spikes = trial.spikes_smth.loc[trial.cond_id.n >= 0]
                    if smth_spikes.shape[0] == trial_len:
                        smth_trial_list.append(smth_spikes.to_numpy()[:, dataset.heldin_channels])

            # take the mean of all trials in condition
            smth_trial_list = np.array(smth_trial_list)
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
            this_dataset_centered = (this_dataset_data.T - this_dataset_means.T)

            # run Ridge regression to fit this session to dim reduced data
            reg = Ridge(alpha=1.0, fit_intercept=False)
            reg.fit(this_dataset_centered, dim_reduced_data_this)

            # use the coefficients as the alignment matrix
            matrix = torch.from_numpy(np.copy(reg.coef_.astype(np.float32)))
            alignment_matrices[session] = matrix  # n_chans x n_PCs

            # mean subtract the data after the readin using the bias
            bias = torch.from_numpy((-1 * np.dot(this_dataset_means, reg.coef_.T)).astype(np.float32))
            alignment_biases[session] = bias # n_chans

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
    alignment_matrices, alignment_biases = {}, {}
    with h5py.File(path, 'r') as h5:
            for session in config.data.sessions:
                alignment_matrices[session] = torch.Tensor(np.array(h5[session]['matrix']))
                alignment_biases[session] = torch.Tensor(np.array(h5[session]['bias']))
    return alignment_matrices, alignment_biases
datasets = {}


def get_alignment_matricies(config):
    ''' 
    '''
    # # Cache datasets for later use
    # cached_data_dir = osp.join(config.dirs.dataset_dir, 'cached_data')
    # os.makedirs(cached_data_dir, exist_ok=True)
    
    # # datasets = {}
    # # for session in config.data.sessions:
    # #     filename = get_data_filename(config, session)
    # #     cached_data_path = osp.join(cached_data_dir, filename)

    # #     dataset = get_toolkit_dataset(config, session)

    # #     datasets[session] = dataset

    # datasets = load_toolkit_datasets(config)


    # session_csv = pd.read_csv(f'sessions.csv')

    # avg_conds = {}
    # trials_dict = {}

    # for session in config.data.sessions:
    #     dataset = copy.deepcopy(datasets[session])

    #     # if config.data.rem_xcorr: 
    #     #     dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)

    #     # dataset.resample(config.data.bin_size / 1000)
    #     # dataset.smooth_spk(config.data.smth_std, name='smth')

    #     # speed = np.linalg.norm(dataset.data.decVel, axis=1)
    #     # dataset.data['speed'] = speed
        
    #     # dataset.calculate_onset('speed', onset_threshold=0.005)

    #     failed_trials = ~dataset.trial_info['is_successful'] 
    #     center_trials = dataset.trial_info['is_center_target']
    #     ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
    #     cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()
        
    #     trial_data = dataset.make_trial_data(
    #         align_field='speed_onset',
    #         align_range=(-350, 1250),
    #         allow_overlap=True,
    #         ignored_trials=failed_trials | cl_blocks
    #     )

    #     trial_data.sort_index(axis=1, inplace=True)
    #     n_channels = trial_data.spikes.shape[-1]

    #     n_heldout = int(config.train.pct_heldout * n_channels)
    #     np.random.seed(config.train.heldout_seed)
    #     heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
    #     heldin_channels = torch.ones(n_channels, dtype=bool)
    #     heldin_channels[heldout_channels] = False

    #     avg_conds[session] = []
    #     for cond_id in range(1,9):
    #         smth_trial_list = []
    #         for trial_id, trial in trial_data.groupby('trial_id'):
    #             if datasets[session].trial_info.loc[trial_id].cond_id == cond_id:
    #                 smth_trial_list.append(trial.spikes_smth.to_numpy()[:, heldin_channels])

    #         # take the mean of all trials in condition
    #         smth_trial_list = np.array(smth_trial_list)
    #         cond_avg_trials = np.mean(smth_trial_list, 0)
    #         avg_conds[session].append(cond_avg_trials)

    # session_list = list(avg_conds.keys())

    # avg_cond_arr = np.array(list(avg_conds.values())) # (days, conds, bins, chans)
    # avg_cond_arr = avg_cond_arr.transpose((3, 0, 1, 2)) # -> (chans, days, conds, bins)
    # nchans, ndays, nconds, nbins = avg_cond_arr.shape
    # avg_cond_arr = avg_cond_arr.reshape((nchans * ndays, nconds * nbins))

    # avg_cond_means = avg_cond_arr.mean(axis=1)
    # avg_cond_centered = (avg_cond_arr.T - avg_cond_means.T).T

    # pca = PCA(n_components=config.model.factor_dim)
    # pca.fit(avg_cond_centered.T)

    # dim_reduced_data = np.dot(avg_cond_centered.T, pca.components_.T).T
    # avg_cond_arr = avg_cond_arr.reshape((nchans, ndays, nconds, nbins))

    # dim_reduced_data_means = dim_reduced_data.mean(axis=1)
    # dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T).T

    # alignment_matrices = {}
    # alignment_biases =  {}

    # for day in range(ndays):
    #     this_dataset_data = avg_cond_arr.reshape((nchans, ndays, nconds * nbins))[:, day, :].squeeze()

    #     this_dataset_means = avg_cond_means.reshape(nchans, ndays)[:, day].squeeze()
    #     this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

    #     reg = Ridge(alpha=1.0, fit_intercept=False)
    #     reg.fit(this_dataset_centered.T, dim_reduced_data_this.T)

    #     matrix = torch.from_numpy(np.copy(reg.coef_.astype(np.float32)))  # nchans x nPCs
    #     alignment_matrices[session_list[day]] = matrix
    #     bias = -1 * np.dot(this_dataset_means, reg.coef_.T)
    #     bias = torch.from_numpy(bias.astype(np.float32))
    #     alignment_biases[session_list[day]] = bias


    # return alignment_matrices, alignment_biases
    
    # ''' OLD SECTION'''

    # readins will be randomly initialized
    if config.model.readin_init is None:
        return None, None

    # if initializing a model to fine tune then load in reduced_dim data to align to
    if config.dirs.trained_mdl_path != '':
        return align_new_session(config)

    if config.data.cache_pcr:
        # get directory where cached .h5 files are
        cached_pcr_dir = osp.join(config.dirs.dataset_dir, 'cached_pcr')
        os.makedirs(cached_pcr_dir, exist_ok=True)

        # get filename of cached pcr with same parameters
        h5_filename = get_alignment_filename(config)

        # combine directory and filename
        cached_pcr_path = osp.join(cached_pcr_dir, h5_filename)

        # if cached version of pcr exists, then load it in
        if osp.exists(cached_pcr_path):
            print('\nCached PCR found, Loading in...')

            return load_alignment_matricies(config, cached_pcr_path)

        print('\nCached PCR not found, Creating...')

    else:
        print('\nCreating PCR readins...')
    
    # create cached alignment matrices and return them
    return align_sessions(config)


def get_alignment_filename(config):

    data = config.data
    train = config.train
    model = config.model

    param_list = [
        model.readin_init,
        model.factor_dim, 
        train.pct_heldout,
        train.heldout_seed,
        data.smth_std, 
        data.ol_align_field, 
        data.cl_align_field, 
        *data.ol_align_range,
        *data.cl_align_range,
        *data.sessions
    ]

    param_string = ''.join(f'{param}' for param in param_list)

    hashed_str = hashlib.md5(param_string.encode()).hexdigest()

    h5_filename = f'pcr_{hashed_str}.h5'

    return  h5_filename