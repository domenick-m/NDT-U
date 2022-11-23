import pickle as pkl
import os.path as osp
from data.t5_dataset import T5CursorDataset
import torch
import os
import numpy as np
import copy
import pandas as pd
from utils.data_utils import chop

def load_toolkit_datasets(config):
    ''' TODO
    switch to osp join
    '''
    os.makedirs(config.dirs.proc_data_dir, exist_ok=True)
    datasets = {}
    for session in config.data.sessions:
        if osp.exists(f'{config.dirs.proc_data_dir}/{session}.pkl'):
             with open(f'{config.dirs.proc_data_dir}/{session}.pkl', "rb") as ifile:
                dataset = pkl.load(ifile)
        else:
            dataset = T5CursorDataset(f'{config.dirs.raw_data_dir}/{session}.mat')
            with open(f'{config.dirs.proc_data_dir}/{session}.pkl', 'wb') as tfile:
                pkl.dump(dataset, tfile)

        datasets[session] = dataset
    return datasets


def get_pretraining_data(config):
    ''' 
    TODO
    Mke print done after finished and show dataset size
    add a cached dataobject and a txt file to know what the config was if same , use else make new and cache
    '''
    print('\nLoading in Data...\n')
    datasets = load_toolkit_datasets(config)

    chopped_spikes, session_names = [], []
    for session in config.data.sessions:
        dataset = datasets[session]

        if config.data.rem_xcorr: 
            dataset.get_pair_xcorr('spikes', threshold=config.data.xcorr_thesh, zero_chans=True)

        dataset.resample(config.data.bin_size / 1e3) # convert ms to sec

        block_spikes = []
        for block_num, block in dataset.data.groupby(('blockNums', 'n')):
            block_spikes.append(chop(block.spikes.to_numpy(), config.data.seq_len, config.data.overlap))
        spikes = np.concatenate(block_spikes, 0)

        names = [session for i in range(spikes.shape[0])]

        chopped_spikes.append(torch.from_numpy(spikes.astype(np.float32)))
        session_names.append(names)

    return torch.cat(chopped_spikes, 0), np.concatenate(session_names, 0)