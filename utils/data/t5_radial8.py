import pickle as pkl
import os.path as osp
from data.t5_dataset import T5CursorDataset

def load_toolkit_datasets(sessions, processed_data_dir):
    ''' TODO
    '''
    datasets = {}
    raw_data_dir = '/snel/share/share/data/bg2/t5_cursor'
    for session in sessions:
        if osp.exists(f'{processed_data_dir}/{session}.pkl'):
             with open(f'{processed_data_dir}/{session}.pkl', "rb") as ifile:
                dataset = pkl.load(ifile)
        else:
            dataset = T5CursorDataset(f'{raw_data_dir}/{session}.mat')
            with open(f'{processed_data_dir}/{session}.pkl', 'wb') as tfile:
                pkl.dump(dataset, tfile)
        datasets[session] = dataset
    return datasets

def get_pretraining_data(config):
    ''' 
    '''
    # Cache datasets for later use
    load_toolkit_datasets(config.data.pretrain_sessions,)

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