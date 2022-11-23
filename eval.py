import torch
from utils.t5_utils import load_toolkit_datasets
import pandas as pd
import copy
import os.path as osp
import numpy as np
from utils.training_utils import set_seeds, set_device
from utils.config_utils import get_config_from_file
from transformer_ import Transformer
import pickle as pkl
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from utils.data_utils import chop
from utils.plotting.plot_pca import plot_pca


def run_from_path(path='/home/dmifsud/Projects/NDT-U/runs/test/test_4/last.pt'):
    run_dir = osp.dirname(path)
    run_name = osp.basename(run_dir)

    config = get_config_from_file(osp.join(run_dir, 'config.yaml'))

    set_seeds(config)
    set_device(config, {})

    with open(osp.join(run_dir, 'dataset.pkl'), "rb") as dfile:
        dataset = pkl.load(dfile)

    model = Transformer(config, dataset).to(torch.device('cuda:0'))
    model.load_state_dict(torch.load(path), strict=False)

    print('Done!')

    run_evaluation(config, model)

def run_evaluation(config, model):
    # turn off dropout
    model.eval()

    # device to run inference on
    device = torch.device('cuda:0')

    # load each snel_toolkit dataset (one per session) into dict
    datasets = load_toolkit_datasets(config)

    trialized_data = {}
    corr_channels = {}
    cond_list = None

    # loop through sessions that model was trained on
    for session in config.data.sessions:
        # pull one session out of the dict
        dataset = datasets[session]
        
        if config.data.rem_xcorr: 
            xcorr_channels = []
            _, corr_chans = dataset.get_pair_xcorr('spikes', threshold=config.data.xcorr_thesh, zero_chans=True)
            for channel in corr_chans:
                xcorr_channels.append(int(channel.replace('ch', '')))
            xcorr_mask = torch.ones(torch_rates.shape[-1], dtype=bool)
            xcorr_mask[xcorr_channels] = False

        dataset.resample(config.data.bin_size / 1000)
        dataset.smooth_spk(config.data.smth_std, name='smth') 
        
        spikes = dataset.data.spikes.to_numpy()
        spikes_idx = dataset.data.spikes.index

        n_channels = spikes.shape[-1]
        n_heldout = int(config.data.pct_heldout * n_channels)
        
        np.random.seed(config.data.heldout_seed)
        heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
        heldin_channels = torch.ones(n_channels, dtype=bool)
        heldin_channels[heldout_channels] = False

        if config.data.rem_xcorr:
            xcorr_hi = xcorr_mask[heldin_channels]
            xcorr_ho = xcorr_mask[heldout_channels]

            corr_channels[session] = {
                'hi': xcorr_hi, 
                'ho': xcorr_ho,
                'all': np.concatenate((xcorr_hi, xcorr_ho), -1)
            }

        chopped_hi_spks = chop(spikes[:, heldin_channels], config.data.seq_len, config.data.seq_len - 1)
        chopped_hi_spks = torch.from_numpy(chopped_hi_spks)

        n_samples = chopped_hi_spks.shape[0]
        names = [session for i in range(n_samples)]

        batch_size = 512

        with torch.no_grad():
            torch_rates, output = [], []

            for i in range(0, n_samples, batch_size):
                i_ = i + batch_size
                ret_tuple = model(chopped_hi_spks[i:i_].to(device), names[i:i_])

                torch_rates.append(ret_tuple[0])
                output.append(ret_tuple[1])

            rates = torch.cat(torch_rates, 0)
            output = torch.cat(output, 0)

        # create rates dataframe
        rates_df = pd.DataFrame(
            rates[:, -1, :].cpu().numpy(), 
            index=spikes_idx[config.data.seq_len - 1:], 
            columns=pd.MultiIndex.from_tuples([('rates', f'{i}') for i in range(rates.shape[-1])])
        )
        dataset.data = pd.concat([dataset.data, rates_df], axis=1)
        dataset.smooth_spk(config['data']['smth_std'], signal_type='rates', name='smth')

        # create factors dataframe
        factors_df = pd.DataFrame(
            output[:, -1, :].cpu().numpy(), 
            index=spikes_idx[config.data.seq_len - 1:], 
            columns=pd.MultiIndex.from_tuples([('factors', f'{i}') for i in range(output.shape[-1])])
        )
        dataset.data = pd.concat([dataset.data, factors_df], axis=1)
        dataset.smooth_spk(config['data']['smth_std'], signal_type='factors', name='smth')

        # trialize data
        trial_data = dataset.make_trial_data(
            align_field='start_time',
            ignored_trials=~dataset.trial_info['is_successful'] 
        )

        trial_data.sort_index(axis=1, inplace=True)
        trial_data['X&Y'] = list(zip(trial_data.targetPos.x, trial_data['targetPos']['y']))
        trial_data['condition'] = 0

        if cond_list == None:
            target_pos = list(trial_data['X&Y'].unique())
            target_pos.insert(0, target_pos.pop(target_pos.index((0,0))))
            cond_list = list(zip(target_pos, np.arange(0,9)))

        for xy, id in cond_list:
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id
    
        trialized_data[session] = trial_data

    factors = []
    for session in config.data.sessions:    
        for ids, trial in trialized_data[session].groupby(['condition', 'trial_id']):
            # do not run pca on return trials
            if ids[0] != 0:
                factors.append(trial.factors_smth.to_numpy())

    factors = np.concatenate(factors, 0)
    pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
    pca.fit(factors)

    cond_avg_low_d_data, cond_avg_ids = [], []
    sing_trial_low_d_data, sing_trial_ids = [], []

    for session in config.data.sessions:    
        for cond_id, trials in trialized_data[session].groupby('condition'):
            low_d_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                low_d_trials.append(pca.transform(trial.factors_smth))
            low_d_trials = np.concatenate(low_d_trials, 0)

            sing_trial_low_d_data.append(low_d_trials)
            sing_trial_ids.append(cond_id)

            cond_avg_low_d_data.append(low_d_trials.mean(0))
            cond_avg_ids.append(cond_id)

    # html_string = plot_pca()

    # with open('test.html', 'a') as f: f.write(hi)

if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        run_from_path()
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')




