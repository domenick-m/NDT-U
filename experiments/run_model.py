from utils.data.create_local_t5data import get_trial_data
from datasets import get_testing_data
from utils_f import get_config
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from utils.plot.plot_true_vs_pred_mvmnt import plot_true_vs_pred_mvmnt
import torch
from utils_f import get_config_from_file, set_seeds, set_device
from datasets import get_trial_data, chop, smooth_spikes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
import sys
import wandb
import pandas as pd
import copy
import pickle
import sys

from data.t5_dataset import T5CursorDataset
  
print("This is the name of the program:", sys.argv[0])
  
print("Argument List:", str(sys.argv))




def run(dataset, config, model, session, device):
    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

    if osp.exists('session_trials/cond_list.pickle'):
        with open('session_trials/cond_list.pickle', 'rb') as handle:
            cond_list = pickle.load(handle)
    else:
        cond_list = None
    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

    if config.data.rem_xcorr: 
        corr, corr_chans = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)
        
    dataset.resample(config.data.bin_size / 1000)
    dataset.smooth_spk(60, name='smth') # for use if we want to take mean and std of smth values

    failed_trials = ~dataset.trial_info['is_successful'] 
    center_trials = dataset.trial_info['is_center_target']
    ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item()
    cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()

    spks = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.to_numpy()
    spks_idx = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.index

    n_channels = dataset.data.spikes.shape[-1]

    n_heldout = int(config.data.heldout_pct * n_channels)
    n_heldin = n_channels - n_heldout
    np.random.seed(config.setup.seed)
    heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
    heldin_channels = torch.ones(n_channels, dtype=bool)
    heldin_channels[heldout_channels] = False

    chopped_spks = chop(np.array(spks[:, heldin_channels]), 30, 29)
    hi_chopped_spks = torch.Tensor(chopped_spks).to(device)


    names = [session for i in range(hi_chopped_spks.shape[0])]
    with torch.no_grad():
        _, output = model(hi_chopped_spks, names)


    # factors_df = pd.DataFrame(outputs[:, -1, :], index=spks_idx[29:], columns=pd.MultiIndex.from_tuples([('factors', f'{i}') for i in range(output.shape[-1])]))
    factors_df = pd.DataFrame(output[:, -1, :].cpu().numpy(), index=spks_idx[29:], columns=pd.MultiIndex.from_tuples([('factors', f'{i}') for i in range(output.shape[-1])]))
    dataset.data = pd.concat([dataset.data, factors_df], axis=1)

    dataset.smooth_spk(config['data']['smth_std'], signal_type='factors', name='smth')

    ignored_trials = failed_trials | center_trials | cl_blocks
    ignored_trials[1] = True

    trial_data = dataset.make_trial_data(
        align_field='start_time',
        align_range=(0, config.data.trial_len),
        allow_overlap=True,
        ignored_trials= ignored_trials
    )

    trial_data.sort_index(axis=1, inplace=True)
    trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
    trial_data['condition'] = 0

    if cond_list == None:
        cond_list = list(zip(trial_data['X&Y'].unique(), np.arange(1,9)))
    for xy, id in cond_list:
        indices = trial_data.index[trial_data['X&Y'] == xy]
        trial_data.loc[indices, 'condition'] = id
        print(id, xy)

    factors = []
    for cond_id, trials in trial_data.groupby('condition'):
        for trial_id, trial in trials.groupby('trial_id'):
            factors.append(trial.factors_smth.to_numpy())

    np.save(f'session_factors/{session}_.npy', np.concatenate(factors, 0))

    with open(f'session_trials/{session}_.pickle', 'wb') as handle:
        pickle.dump(trial_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not osp.exists('session_trials/cond_list.pickle'):  
        with open(f'session_trials/cond_list.pickle', 'wb') as handle:
            pickle.dump(cond_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    session = sys.argv[1]
    path = sys.argv[2]

    dataset = T5CursorDataset(f'/home/dmifsud/Projects/NDT-U/data/{session}.mat')

    name = path[:path.rindex('/')].split('/')[-1]
    config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')

    set_seeds(config)
    set_device(config, {})
    device = torch.device('cuda:0')

    model = torch.load(path).to(device)
    model.name = name

    model.eval()
    run(dataset, config, model, session, device)