import pickle as pkl
import os.path as osp
# from data.t5_radial_8.dataset import T5CursorDataset
import torch
import os
import hashlib

import numpy as np
import copy
import pandas as pd
from utils.data_utils import chop, get_heldin_mask
from utils.eval_utils import chop_infer_join
import sys
from data.t11_piano import dataset as t11_piano
from data.t5_radial_8 import dataset as t5_radial_8
from data.t11_fixed_decoder import dataset as t11_fixed_decoder

def load_toolkit_datasets(config, verbose=False):
    '''
    '''
    n_cached = 0
    n_sessions = len(config.data.sessions)

    if config.data.cache_toolkit:
        # if cached data directory does not exist, create it
        cached_data_dir = osp.join(config.dirs.dataset_dir, 'cached_data')
        os.makedirs(cached_data_dir, exist_ok=True)
    
    elif verbose:
        print(f'\nCreating {n_sessions} snel-toolkit datasets...')

    datasets = {}
    for session in config.data.sessions:
        if config.data.cache_toolkit:
            # cached data should have same parameters as config
            filename = get_data_filename(config, session)
            cached_data_path = osp.join(cached_data_dir, filename)

            # if dataset is cached, load it in
            if osp.exists(cached_data_path):
                with open(cached_data_path, "rb") as ifile:
                    dataset = pkl.load(ifile)
                    n_cached += 1

        # load mat into snel_toolkit dataset object
        dataset = get_toolkit_dataset(config, session)

        n_channels = dataset.data.spikes.shape[-1]        

        # if masking correlated channels, create a mask of the non-correled channels
        if config.data.rem_xcorr: 
            # use snel_toolkit function to remove correlated channels
            _, corr_chans = dataset.get_pair_xcorr('spikes', threshold=config.data.xcorr_thesh, zero_chans=True)

            # create the mask, init to all True
            xcorr_mask = torch.ones(n_channels, dtype=bool)

            # corr_chans is the dataframe column names with 'ch' prefix, convert them to indexs
            xcorr_channels = []
            for channel in corr_chans:
                xcorr_channels.append(int(channel.replace('ch', '')))

            # indicate where correlated channels are with False
            xcorr_mask[xcorr_channels] = False

        # convert bin size from ms to sec 
        dataset.resample(config.data.bin_size / 1e3) 
        dataset.smooth_spk(config.data.smth_std, name='smth')
        dataset.data = dataset.data.copy()

        # only t5 radial8 has decVel
        if 'decVel' in dataset.data:
            # calculate speed from X and Y velocity
            speed = np.linalg.norm(dataset.data.decVel, axis=1)
            dataset.data['speed'] = speed
            
            # calculate movement onset with default values
            dataset.calculate_onset('speed', onset_threshold=0.005)

        # get heldin channels and store on dataset
        dataset.heldin_channels = get_heldin_mask(config, n_channels)

        # create the final correlated channel mask to apply to rates
        if config.data.rem_xcorr:
            # get an xcorr_mask for the heldin and heldout channels
            xcorr_hi = xcorr_mask[dataset.heldin_channels]
            xcorr_ho = xcorr_mask[~dataset.heldin_channels]

            # stack heldin and heldout to arrange channels like rates
            dataset.xcorr_mask = np.concatenate((xcorr_hi, xcorr_ho), -1)

        if config.data.cache_toolkit:
            # cache processed dataset
            with open(cached_data_path, 'wb') as tfile:
                pkl.dump(dataset, tfile)

        # store dataset object in session dictionary
        datasets[session] = dataset

    if config.data.cache_toolkit and verbose:
        n_created  = n_sessions - n_cached

        print((
            f'\nLoaded in {n_cached} cached snel-toolkit datasets.' if n_cached > 0 else ''
            f'\nCreated and cached {n_created} snel-toolkit datasets.' if n_created > 0 else ''
        ))

    # return session dictionary
    return datasets


def get_pretraining_data(config):
    ''' 
    TODO
    Mke print done after finished and show dataset size
    add a cached dataobject and a txt file to know what the config was if same , use else make new and cache
    '''
    datasets = load_toolkit_datasets(config, verbose=True)

    chopped_spikes, session_names = [], []
    for session in config.data.sessions:
        spikes_arr = []
        # chop spikes within each block to avoid overlapping blocks 
        for _, block in datasets[session].data.groupby(('block_num', 'n')):
            if config.data.use_cl or block.trial_type.to_numpy()[0,0] == 'OL':
                if block.spikes.to_numpy().shape[0] >= config.data.seq_len: 
                    spikes_arr.append(chop(block.spikes.to_numpy(), config.data.seq_len, config.data.overlap))

        assert len(spikes_arr) != 0, 'Make sure data has open-loop control if data.use_cl is False.' # TODO

        spikes_arr = np.concatenate(spikes_arr, 0)
        chopped_spikes.append(torch.from_numpy(spikes_arr.astype(np.float32)))
        session_names.append([session for i in range(spikes_arr.shape[0])])

    # return chopped spikes tensor and name array
    return torch.cat(chopped_spikes, 0), np.concatenate(session_names, 0)


def get_trialized_data(config, datasets, model=None):
    '''
    '''
    trialized_data = {}

    for session in config.data.sessions:
        trialized_data[session] = {}

        # get snel_toolkit dataset from load_toolkit_dataset's dictionary
        dataset = datasets[session]

        # if model included in call then run inference and before making trials
        if model is not None:
            dataset = chop_infer_join(config, dataset, session, model)

        # load in sessions.csv to get the open- and closed-loop blocks
        cl_blocks =  dataset.trial_info['trial_type'].isin(['CL']).values.squeeze()

        # trialize open-loop data
        if dataset.has_OL:
            trialized_data[session]['ol_trial_data'] = dataset.make_trial_data(
                align_field=config.data.ol_align_field,
                align_range=(config.data.ol_align_range[0], config.data.ol_align_range[1]),
                ignored_trials=~dataset.trial_info['is_successful'] | cl_blocks,
                allow_overlap=True
            )

        # trialize closed-loop data
        if dataset.has_CL:
            trialized_data[session]['cl_trial_data'] = dataset.make_trial_data(
                align_field=config.data.cl_align_field,
                align_range=(config.data.cl_align_range[0], config.data.cl_align_range[1]),
                # ignored_trials= ~cl_blocks,
                ignored_trials=~dataset.trial_info['is_successful'] | ~cl_blocks,
                allow_overlap=True
            )

    return trialized_data

def get_data_filename(config, session):

    dirs = config.dirs
    data = config.data
    train = config.train

    param_list = [
        session,
        dirs.dataset_dir,
        data.rem_xcorr,
        data.xcorr_thesh, 
        data.bin_size, 
        data.smth_std, 
        train.pct_heldout, 
        train.heldout_seed
    ]

    param_string = ''.join(f'_{param}' for param in param_list)

    hashed_str = hashlib.md5(param_string.encode()).hexdigest()

    h5_filename = f'data_{hashed_str}.pkl'

    return  h5_filename

def get_toolkit_dataset(config, session):
    sys.path.append(config.dirs.dataset_dir)
    from dataset import init_toolkit_dataset
    return init_toolkit_dataset(session)