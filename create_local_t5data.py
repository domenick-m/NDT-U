from data.t5_dataset import T5CursorDataset
import numpy as np
import copy
import pandas as pd

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


def get_training_data(config):
    ''' Get spikes to train NDT with
    '''
    # Cache datasets for test data
    populate_datasets(config)

    chopped_spikes, session_names = [], []
    for session in config.data.pretrain_sessions:
        dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

        if config.data.rem_xcorr: 
            dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)

        dataset.resample(config.data.bin_size / 1000) # convert ms to sec
        spikes = dataset.data.spikes.to_numpy()
        spikes = chop(spikes, config.data.seq_len, config.data.overlap)
        names = [session for i in range(spikes.shape[0])]

        chopped_spikes.append(spikes)
        session_names.append(names)

    return np.concatenate(chopped_spikes, 0), np.concatenate(session_names, 0)


def get_trial_data(config, smth_list=None, max_lag=None):
    ''' Get trialized data for test evaluation
    '''
    # Cache datasets for test data
    populate_datasets(config)
    lag = config.data.lag if max_lag == None else max_lag

    def trials_from_list(session_list):
        ''' Helper function to take in list from config and return dict
        '''
        trials = {}

        def add_trials(dataset, control, blocks):
            ''' Helper function to make trial data and add conditions 
            '''
            failed_trials = ~dataset.trial_info['is_successful'] 
            center_trials = dataset.trial_info['is_center_target'] if not config.data.center_trials else False
            
            trial_data = dataset.make_trial_data(
                align_field='start_time',
                align_range=(
                    -config.data.seq_len * config.data.bin_size + config.data.bin_size, # start align
                    config.data.trial_len + lag if control == 'ol_blocks' else None # end_align
                ),
                allow_overlap=True,
                ignored_trials=failed_trials | center_trials
            )

            trial_data.sort_index(axis=1, inplace=True)
            trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
            trial_data['condition'] = 0

            for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):    
                indices = trial_data.index[trial_data['X&Y'] == xy]
                trial_data.loc[indices, 'condition'] = id

            trials[session][control] = {}
            for block in blocks:
                trials[session][control][block] = {}
                block_mask = trial_data['blockNums'].isin([block]).values.squeeze()
                for tr_id, trial in trial_data[block_mask].groupby('trial_id'):
                    trials[session][control][block][tr_id] = trial

        session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

        for session in block_list:
            trials[session] = {}

            dataset = datasets[session]
            if smth_list != None and not f'spikes_smth_{smth_list[0]}' in dataset.data:
                dataset.resample(config.data.bin_size / 1000)
                # Running a sweep for lag & smoothing
                for std in smth_list:
                    dataset.smooth_spk(std, name=f'smth_{std}')
            elif smth_list == None and not 'spikes_smth' in dataset.data:
                dataset.resample(config.data.bin_size / 1000) # convert ms to sec
                # Normal test eval
                dataset.smooth_spk(config.data.smth_std, name='smth')

            add_trials(dataset, 'ol_blocks', ol_blocks)
            add_trials(dataset, 'cl_blocks', cl_blocks)

        return trials

    train_trials = trials_from_list(config.data.pretrain_sessions)
    # test_trials = trials_from_list(config.data.test)

    # return train_trials, test_trials
    return train_trials, None