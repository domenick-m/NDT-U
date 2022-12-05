from utils.data_utils import chop, smooth
import torch
import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

def merge_with_df(config, data, idx, name, dataset, smooth_data=True):
    if smooth_data:
        data = smooth(data, config.data.smth_std, config.data.bin_size, causal=False)
        name = f'{name}_smth'
    col_labels = pd.MultiIndex.from_tuples([(name, f'{i}') for i in range(data.shape[-1])])
    smth_df = pd.DataFrame(data, index=idx, columns=col_labels)
    dataset.data = pd.concat([dataset.data, smth_df], axis=1)
    return dataset

def chop_infer_join(config, dataset, session, model):
    # extract spikes and indices to join factors / rates
    spikes = dataset.data.spikes.to_numpy()
    spikes_idx = dataset.data.spikes.index

    # chop spikes and create list of session names 
    chopped_hi_spks = chop(spikes[:, dataset.heldin_channels], config.data.seq_len, config.data.seq_len - 1)
    chopped_hi_spks = torch.from_numpy(chopped_hi_spks)
    n_samples = chopped_hi_spks.shape[0]
    names = [session for i in range(n_samples)]

    # run batched inference
    batch_size = 512
    with torch.no_grad():
        rates, output = [], []
        for i in range(0, n_samples, batch_size):
            i_ = i + batch_size
            b_rates, b_output= model(chopped_hi_spks[i:i_].to(torch.device('cuda:0')), names[i:i_])

            rates.append(b_rates)
            output.append(b_output)

        rates = torch.cat(rates, 0)
        output = torch.cat(output, 0)

    # create rates dataframe
    np_output = rates[:, -1, :].cpu().numpy()
    dataset = merge_with_df(config, np_output, spikes_idx[config.data.seq_len - 1:], 'rates', dataset, False)
    dataset = merge_with_df(config, np_output, spikes_idx[config.data.seq_len - 1:], 'rates', dataset)

    # create factors dataframe
    np_output = output[:, -1, :].cpu().numpy()
    dataset = merge_with_df(config, np_output, spikes_idx[config.data.seq_len - 1:], 'factors', dataset, False)
    dataset = merge_with_df(config, np_output, spikes_idx[config.data.seq_len - 1:], 'factors', dataset)

    # return merged dataset
    return dataset

def run_pca(config, trialized_data, pca):
    '''
    '''
    # tuple is composed of: (data_list, cond_id_list)
    ol_cond_avg = ([], []) 
    ol_single_trial = ([], [])
    cl_single_trial = ([], [])

    # make sure that each trial is this long
    trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size

    for session in config.data.sessions:    
        for cond_id, trials in trialized_data[session]['ol_trial_data'].groupby(('cond_id', 'n')):
            if cond_id > 0:
                low_d_trials = []
                for trial_id, trial in trials.groupby('trial_id'):
                    if trial.factors_smth.shape[0] == trial_len:
                        low_d_trials.append(pca.transform(trial.factors_smth.to_numpy()))

                ol_single_trial[0].append(np.concatenate(low_d_trials, 0))
                ol_single_trial[1].append(cond_id)

                ol_cond_avg[0].append(np.array(low_d_trials).mean(0))
                ol_cond_avg[1].append(cond_id)

        for cond_id, trials in trialized_data[session]['cl_trial_data'].groupby(('cond_id', 'n')):
            if cond_id > 0:
                low_d_trials = []
                for trial_id, trial in trials.groupby('trial_id'):
                    low_d_trials.append(pca.transform(trial.factors_smth.to_numpy()))

                cl_single_trial[0].append(np.concatenate(low_d_trials, 0))
                cl_single_trial[1].append(cond_id)
    
    return ol_cond_avg, ol_single_trial, cl_single_trial

def run_decoding(config, trialized_data):
    '''
    '''
    lag_bins = int(config.data.lag / config.data.bin_size)

    all_vel, all_rates, all_factors = [], [], []
    for session in config.data.sessions:
        for cond_id, trials in trialized_data[session]['ol_trial_data'].groupby('condition'):
            if cond_id != 0:
                for trial_id, trial in trials.groupby('trial_id'):
                    all_vel.append(trial.decVel.to_numpy()[lag_bins:])
                    all_rates.append(trial.rates_smth.to_numpy()[:-lag_bins])
                    all_factors.append(trial.factors_smth.to_numpy()[:-lag_bins])
    
    all_vel = np.concatenate(all_vel, 0)
    all_rates = np.concatenate(all_rates, 0)
    all_factors = np.concatenate(all_factors, 0)

    result_dict = {}

    rates_decoder = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    rates_decoder.fit(all_rates, all_vel)
    result_dict['rates_decoding_all'] = rates_decoder.score(all_rates, all_vel)

    factors_decoder = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    factors_decoder.fit(all_factors, all_vel)
    result_dict['factors_decoding_all'] = factors_decoder.score(all_factors, all_vel)

    # if config.log.to_wandb:
        # wandb.log(result_dict)

    if config.log.to_csv:
        with open(f'{config.dirs.save_dir}/log.csv', 'a') as f:
            results = '\n'
            for k, v in result_dict:
                results += f'[{k}: {v}] '
            f.write(results)

