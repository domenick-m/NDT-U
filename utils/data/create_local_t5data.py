from data.t5_dataset import T5CursorDataset
import numpy as np
import copy

def chop(data, seq_len, overlap):
    shape = (int((data.shape[0] - overlap) / (seq_len - overlap)), seq_len, data.shape[-1])
    strides = (data.strides[0] * (seq_len - overlap), data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape, strides).copy().astype('f')

datasets = {}

def populate_datasets(config):
    ''' Create dataset object for each session and store'''
    if datasets == {}:
        for session, _ in [*config.data.train, *config.data.test]:
            if not session in datasets:
                datasets[session] = T5CursorDataset(f'{config.data.dir}/{session}.mat')


def get_training_data(config):
    ''' Get spikes to train NDT with'''
    # Cache datasets for test data
    populate_datasets(config)

    chopped_blocks = []
    for session, ol_blocks, cl_blocks in config.data.train:
        dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

        if config.data.rem_xcorr: 
            dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)

        dataset.resample(config.data.bin_size / 1000) # convert ms to sec
        data = dataset.data
        for block in ol_blocks+cl_blocks:
            block_data = data[data['blockNums'].isin([block]).values.squeeze()]
            spikes = block_data.spikes.to_numpy()
            chopped_blocks.append(chop(spikes, config.data.seq_len, config.data.overlap))

    return np.concatenate(chopped_blocks, 0)

def get_blocks(config, block_list):
    trials = {}
    for session, blocks in block_list:
        trials[session] = {}

        dataset = datasets[session]
        dataset.resample(config.data.bin_size / 1000) # convert ms to sec

        failed_trials = ~dataset.trial_info['is_successful'] if config.data.failed_trials else None
        center_trials = dataset.trial_info['is_center_target'] if config.data.center_trials else None

        trial_data = dataset.make_trial_data(
            align_field='start_time',
            align_range=(
                config.data.seq_len * config.data.bin_size, # start align
                config.data.trial_len + config.data.lag # end align
            ),
            allow_overlap=True,
            ignored_trials=np.all((failed_trials, center_trials), 0)
        )

        trial_data.sort_index(axis=1, inplace=True)
        trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
        trial_data['condition'] = 0

        for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):    
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id
        
        for block in blocks:
            trials[session][block] = {}





def get_trial_data(config):
    ''' Get trialized data for test evaluation'''
    # Cache datasets for test data
    populate_datasets(config)

    test_trials = {}
    

        
            