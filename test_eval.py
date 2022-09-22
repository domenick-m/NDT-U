import os
import sys
import copy
import h5py
import torch
import torch.nn as nn
import shutil
import warnings
import wandb
import math
import torch
import torch.nn as nn
import fileinput
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.linalg import LinAlgWarning
# from create_local_data import make_test_data
from nlb_tools.make_tensors import h5_to_dict
# from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.evaluation import bits_per_spike
from sklearn.model_selection import GridSearchCV
from utils import get_config_from_file, set_seeds
from sklearn.model_selection import KFold

from sklearn.linear_model import PoissonRegressor

from plot_utils.plot_rates_vs_spks_indv import plot_rates_vs_spks_indv
# from plot_utils.plot_rates_vs_spks_all import plot_rates_vs_spks_all
from plot_utils.plot_pca import plot_pca
# from plot_utils.plot_true_vs_pred_mvmnt import plot_true_vs_pred_mvmnt
from datasets import get_dataloaders
import multiprocessing
import scipy.signal as signal

warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')


# Main function should have:
    #  set mode (list of files)
    #  single mode (single run name)

def smooth_spikes(data, gauss_width, bin_width, causal):
    kern_sd = int(gauss_width / bin_width)
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    if causal: 
        window[len(window) // 2:] = 0
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    return np.apply_along_axis(filt, 0, data)

def norm_rates(rates):
    rates_new = copy.deepcopy(rates)
    for ch in range(rates.shape[1]):
        mean = rates[:, ch].mean()
        std = rates[:, ch].std()
        rates_new[:, ch] -= mean
        rates_new[:, ch] /= std
    return rates_new

def run_pca(normed_rates):
    tr_list = []
    pca = PCA(n_components=3)
    pca.fit(np.concatenate(normed_rates, axis=0))
    for trial in normed_rates:
        tr_list.append(pca.transform(trial))
    return tr_list

def eval_mc_rtt(config, model, local_save):
    path = '{0}{1}_{2}_{3}_{4}.h5'.format(
        config["setup"]["data_dir"], 
        config["setup"]["dataset"], 
        config["train"]["seq_len"], 
        config["train"]["overlap"], 
        config["train"]["lag"]
    )

    model.eval()

    device = torch.device('cuda:0')
    set_seeds(config)

    with h5py.File(path, 'r') as h5file:
        h5dict = h5_to_dict(h5file)

    def run_exclusion_trials(trial_idx, trial_hi, trial_ho):
        trial_rates, normed_rates = [], []
        trial_c_smth_rates, normed_c_smth_rates = [], []
        trial_ac_smth_rates, normed_ac_smth_rates = [], []

        indicies = h5dict[trial_idx]
        spikes_list = np.split(h5dict[trial_hi], indicies, axis=0)
        heldout_spikes_list = np.split(h5dict[trial_ho], indicies, axis=0)

        for spikes, heldout_spikes in zip(spikes_list, heldout_spikes_list):
            spikes = torch.Tensor(spikes).to(device)
            heldout_spikes = torch.Tensor(heldout_spikes).to(device)
            spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
            
            with torch.no_grad():
                rates = model(spikes)[:, -1, :].exp().cpu().numpy()
                trial_rates.append(rates)
                trial_c_smth_rates.append(smooth_spikes(rates, config['train']['smth_std'], 10, causal=True))
                trial_ac_smth_rates.append(smooth_spikes(rates, config['train']['smth_std'], 10, causal=False))

                normed_rates.append(norm_rates(rates))
                normed_c_smth_rates.append(norm_rates(smooth_spikes(rates, config['train']['smth_std'], 10, causal=True)))
                normed_ac_smth_rates.append(norm_rates(smooth_spikes(rates, config['train']['smth_std'], 10, causal=False)))

            del spikes
            del heldout_spikes
            torch.cuda.empty_cache()

        trial_pcs = run_pca(normed_rates)
        trial_c_smth_pcs = run_pca(normed_c_smth_rates)
        trial_ac_smth_pcs = run_pca(normed_ac_smth_rates)

        return (
            trial_rates, 
            trial_c_smth_rates, 
            trial_ac_smth_rates, 
            trial_pcs, 
            trial_c_smth_pcs, 
            trial_ac_smth_pcs
        )

    ### DATA ###

    batch_size = config['train']['e_batch_size']

    ## Full Train Dataset
    spikes = torch.Tensor(h5dict['all_train_spikes_heldin']).to(device)
    heldout_spikes = torch.Tensor(h5dict['all_train_spikes_heldout']).to(device)
    with torch.no_grad():
        spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
        num_batches = math.ceil(spikes.size()[0]/batch_size)
        train_rates = []
        for batch in [spikes[batch_size*y:batch_size*(y+1),:,:] for y in range(num_batches)]:
            train_rates.append(model(batch)[:, -1, :].exp().cpu().numpy())
    del spikes
    del heldout_spikes
    torch.cuda.empty_cache()
    train_rates = np.concatenate(train_rates, 0)
    acaus_smth_train_rates = smooth_spikes(train_rates, config['train']['smth_std'], 10, causal=False)
    caus_smth_train_rates = smooth_spikes(train_rates, config['train']['smth_std'], 10, causal=True)

    ## Full Test Dataset
    spikes = torch.Tensor(h5dict['test_spikes_heldin']).to(device)
    heldout_spikes = torch.Tensor(h5dict['test_spikes_heldout']).to(device)
    with torch.no_grad():
        spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)], -1)
        num_batches = math.ceil(spikes.size()[0]/batch_size)
        test_rates = []
        for batch in [spikes[batch_size*y:batch_size*(y+1),:,:] for y in range(num_batches)]:
            test_rates.append(model(batch)[:, -1, :].exp().cpu().numpy())
    del spikes
    del heldout_spikes
    torch.cuda.empty_cache()
    test_rates = np.concatenate(test_rates, 0)
    acaus_smth_test_rates = smooth_spikes(test_rates, config['train']['smth_std'], 10, causal=False)
    caus_smth_test_rates = smooth_spikes(test_rates, config['train']['smth_std'], 10, causal=True)

    ##All Train Trials
    res_tuple = run_exclusion_trials('all_train_trial_idx', 'all_train_trial_hi', 'all_train_trial_ho')
    all_train_trial_rates = res_tuple[0]
    all_train_trial_c_smth_rates = res_tuple[1]
    all_train_trial_ac_smth_rates = res_tuple[2]
    all_train_trial_pcs = res_tuple[3]
    all_train_trial_c_smth_pcs = res_tuple[4]
    all_train_trial_ac_smth_pcs = res_tuple[5]
    
    # All Test Trials
    res_tuple = run_exclusion_trials('all_test_trial_idx', 'all_test_trial_hi', 'all_test_trial_ho')
    all_test_trial_rates = res_tuple[0]
    all_test_trial_c_smth_rates = res_tuple[1]
    all_test_trial_ac_smth_rates = res_tuple[2]
    all_test_trial_pcs = res_tuple[3]
    all_test_trial_c_smth_pcs = res_tuple[4]
    all_test_trial_ac_smth_pcs = res_tuple[5]

    ## LE Train Trials
    res_tuple = run_exclusion_trials('le_train_trial_idx', 'le_train_trial_hi', 'le_train_trial_ho')
    le_train_trial_rates = res_tuple[0]
    le_train_trial_c_smth_rates = res_tuple[1]
    le_train_trial_ac_smth_rates = res_tuple[2]
    le_train_trial_pcs = res_tuple[3]
    le_train_trial_c_smth_pcs = res_tuple[4]
    le_train_trial_ac_smth_pcs = res_tuple[5]

    # LE Test Trials
    res_tuple = run_exclusion_trials('le_test_trial_idx', 'le_test_trial_hi', 'le_test_trial_ho')
    le_test_trial_rates = res_tuple[0]
    le_test_trial_c_smth_rates = res_tuple[1]
    le_test_trial_ac_smth_rates = res_tuple[2]
    le_test_trial_pcs = res_tuple[3]
    le_test_trial_c_smth_pcs = res_tuple[4]
    le_test_trial_ac_smth_pcs = res_tuple[5]

    ## ME Train Trials
    res_tuple = run_exclusion_trials('me_train_trial_idx', 'me_train_trial_hi', 'me_train_trial_ho')
    me_train_trial_rates = res_tuple[0]
    me_train_trial_c_smth_rates = res_tuple[1]
    me_train_trial_ac_smth_rates = res_tuple[2]
    me_train_trial_pcs = res_tuple[3]
    me_train_trial_c_smth_pcs = res_tuple[4]
    me_train_trial_ac_smth_pcs = res_tuple[5]

    # ME Test Trials
    res_tuple = run_exclusion_trials('me_test_trial_idx', 'me_test_trial_hi', 'me_test_trial_ho')
    me_test_trial_rates = res_tuple[0]
    me_test_trial_c_smth_rates = res_tuple[1]
    me_test_trial_ac_smth_rates = res_tuple[2]
    me_test_trial_pcs = res_tuple[3]
    me_test_trial_c_smth_pcs = res_tuple[4]
    me_test_trial_ac_smth_pcs = res_tuple[5]

    ## HE Train Trials
    res_tuple = run_exclusion_trials('he_train_trial_idx', 'he_train_trial_hi', 'he_train_trial_ho')
    he_train_trial_rates = res_tuple[0]
    he_train_trial_c_smth_rates = res_tuple[1]
    he_train_trial_ac_smth_rates = res_tuple[2]
    he_train_trial_pcs = res_tuple[3]
    he_train_trial_c_smth_pcs = res_tuple[4]
    he_train_trial_ac_smth_pcs = res_tuple[5]

    # HE Test Trials
    res_tuple = run_exclusion_trials('he_test_trial_idx', 'he_test_trial_hi', 'he_test_trial_ho')
    he_test_trial_rates = res_tuple[0]
    he_test_trial_c_smth_rates = res_tuple[1]
    he_test_trial_ac_smth_rates = res_tuple[2]
    he_test_trial_pcs = res_tuple[3]
    he_test_trial_c_smth_pcs = res_tuple[4]
    he_test_trial_ac_smth_pcs = res_tuple[5]

    ### DECODING ###

    def get_cobps(rates, spikes):
        return float(bits_per_spike(
            np.expand_dims(np.concatenate(rates, axis=0), axis=1)[:, :, 98:], 
            np.array([i[-1:, :] for i in spikes])
        ))

    result_dict = {}
    result_dict['test co-bps'] = float(bits_per_spike(np.expand_dims(test_rates[:, 98:], axis=1), np.expand_dims(h5dict['test_spikes_heldout'][:, -1, :], axis=1)))
    result_dict['he test trials co-bps'] = get_cobps(he_test_trial_rates, h5dict['he_test_trial_ho'])
    result_dict['me test trials co-bps'] = get_cobps(me_test_trial_rates, h5dict['me_test_trial_ho'])
    result_dict['le test trials co-bps'] = get_cobps(le_test_trial_rates, h5dict['le_test_trial_ho'])

    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(train_rates, h5dict['train_vel'])
    result_dict['test decoding'] = gscv.score(test_rates, h5dict['test_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(caus_smth_train_rates, h5dict['train_vel'])
    result_dict['test causal decoding'] = gscv.score(caus_smth_test_rates, h5dict['test_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(acaus_smth_train_rates, h5dict['train_vel'])
    result_dict['test acausal decoding'] = gscv.score(acaus_smth_test_rates, h5dict['test_vel'])

    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(he_train_trial_rates, axis=0), h5dict['he_train_trial_vel'])
    result_dict['he test decoding'] = gscv.score(np.concatenate(he_test_trial_rates, axis=0), h5dict['he_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(he_train_trial_c_smth_rates, axis=0), h5dict['he_train_trial_vel'])
    result_dict['he test causal decoding'] = gscv.score(np.concatenate(he_test_trial_c_smth_rates, axis=0), h5dict['he_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(he_train_trial_ac_smth_rates, axis=0), h5dict['he_train_trial_vel'])
    result_dict['he test acausal decoding'] = gscv.score(np.concatenate(he_test_trial_ac_smth_rates, axis=0), h5dict['he_test_trial_vel'])
    
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(me_train_trial_rates, axis=0), h5dict['me_train_trial_vel'])
    result_dict['me test decoding'] = gscv.score(np.concatenate(me_test_trial_rates, axis=0), h5dict['me_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(me_train_trial_c_smth_rates, axis=0), h5dict['me_train_trial_vel'])
    result_dict['me test causal decoding'] = gscv.score(np.concatenate(me_test_trial_c_smth_rates, axis=0), h5dict['me_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(me_train_trial_ac_smth_rates, axis=0), h5dict['me_train_trial_vel'])
    result_dict['me test acausal decoding'] = gscv.score(np.concatenate(me_test_trial_ac_smth_rates, axis=0), h5dict['me_test_trial_vel'])
    
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(le_train_trial_rates, axis=0), h5dict['le_train_trial_vel'])
    result_dict['le test decoding'] = gscv.score(np.concatenate(le_test_trial_rates, axis=0), h5dict['le_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(le_train_trial_c_smth_rates, axis=0), h5dict['le_train_trial_vel'])
    result_dict['le test causal decoding'] = gscv.score(np.concatenate(le_test_trial_c_smth_rates, axis=0), h5dict['le_test_trial_vel'])
    gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
    gscv.fit(np.concatenate(le_train_trial_ac_smth_rates, axis=0), h5dict['le_train_trial_vel'])
    result_dict['le test acausal decoding'] = gscv.score(np.concatenate(le_test_trial_ac_smth_rates, axis=0), h5dict['le_test_trial_vel'])


    if not local_save:
        wandb.log(result_dict)
    else:
        with open(f"plots/{model.name}/eval_results.txt", "w") as f:
            for k, v in result_dict:
                f.write(f'{k}: {v}\n')

    ### PLOTS ###

    ## Test Heldin Rates vs Spikes Indv
    # html = plot_rates_vs_spks_indv(
    #     test_rates,
    #     caus_smth_test_rates,
    #     acaus_smth_test_rates,
    #     h5dict['test_hi_30_smth_spikes'],
    #     h5dict['test_hi_50_smth_spikes'],
    #     h5dict['test_hi_80_smth_spikes'],
    #     heldin=True
    # )
    # if not local_save:
    #     wandb.log({'Test Spikes vs Rates Heldin': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/hi_all_spk_vs_rates.html", "w") as f:
    #         f.write(html)

    # ## all_train PCA
    # html = plot_pca(
    #     np.array(all_train_trial_pcs), 
    #     np.array(all_train_trial_c_smth_pcs), 
    #     np.array(all_train_trial_ac_smth_pcs),
    #     h5dict['all_train_trial_angles'], 
    #     config['train']['smth_std'],
    #     add_legend=False,
    #     title='All Train Trials'
    # )
    # if not local_save:
    #     wandb.log({'All Train Rates PCA': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/all_train_rates_pca.html", "w") as f:
    #         f.write(html)

    # ## all_test PCA
    # html = plot_pca(
    #     np.array(all_test_trial_pcs), 
    #     np.array(all_test_trial_c_smth_pcs), 
    #     np.array(all_test_trial_ac_smth_pcs),
    #     h5dict['all_test_trial_angles'], 
    #     config['train']['smth_std'],
    #     add_legend=False,
    #     title='All Test Trials'
    # )
    # if not local_save:
    #     wandb.log({'All Test Rates PCA': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/all_test_rates_pca.html", "w") as f:
    #         f.write(html)

    # ## le_train PCA
    # html = plot_pca(
    #     np.array(le_train_trial_pcs), 
    #     np.array(le_train_trial_c_smth_pcs), 
    #     np.array(le_train_trial_ac_smth_pcs),
    #     h5dict['le_train_trial_angles'], 
    #     config['train']['smth_std'],
    #     add_legend=False,
    #     title='LE Train Trials'
    # )
    # if not local_save:
    #     wandb.log({'LE Train Rates PCA': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/le_train_rates_pca.html", "w") as f:
    #         f.write(html)

    # ## me_train PCA
    # html = plot_pca(
    #     np.array(me_train_trial_pcs), 
    #     np.array(me_train_trial_c_smth_pcs), 
    #     np.array(me_train_trial_ac_smth_pcs),
    #     h5dict['me_train_trial_angles'], 
    #     config['train']['smth_std'],
    #     add_legend=False,
    #     title='ME Train Trials'
    # )
    # if not local_save:
    #     wandb.log({'ME Train Rates PCA': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/me_train_rates_pca.html", "w") as f:
    #         f.write(html)

    # ## he_train PCA
    # html = plot_pca(
    #     np.array(he_train_trial_pcs), 
    #     np.array(he_train_trial_c_smth_pcs), 
    #     np.array(he_train_trial_ac_smth_pcs),
    #     h5dict['he_train_trial_angles'], 
    #     config['train']['smth_std'],
    #     add_legend=False,
    #     title='HE Train Trials'
    # )
    # if not local_save:
    #     wandb.log({'HE Train Rates PCA': wandb.Html(html, inject=False)})
    # else:
    #     with open(f"plots/{model.name}/he_train_rates_pca.html", "w") as f:
    #         f.write(html)


def test_eval(config, model):

    local_save = False
    if not config['wandb']['log']:
        print('W&B Not Enabled; Saving plots locally')
        local_save = True
        if not os.path.isdir(f"plots"): 
            os.makedirs(f"plots")
        if not os.path.isdir(f"plots/{model.name}"): 
            os.makedirs(f"plots/{model.name}")

    if config['setup']['dataset'] == 'mc_rtt':
        eval_mc_rtt(config, model, local_save)
    # else:
    #     eval_t5_cursor(wandb, model, config, device)

def get_lag_and_smth():
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
        exit()
    path = sys.argv[1]
    name = path[:path.rindex('/')].split('/')[-1]
    config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')
    if not os.path.isdir(f"plots/{name}"): os.makedirs(f"plots/{name}")
    shutil.copyfile(path[:path.rindex('/')+1]+'config.yaml', f"plots/{name}/config.yaml")

    path = '{0}{1}_{2}_{3}_{4}.h5'.format(
        config["setup"]["data_dir"], 
        config["setup"]["dataset"], 
        config["train"]["seq_len"], 
        config["train"]["overlap"], 
        config["train"]["lag"]
    )

    device = torch.device('cuda:0')
    set_seeds(config)

    with h5py.File(path, 'r') as h5file:
        h5dict = h5_to_dict(h5file)


    def fit_poisson(alpha, train_x, train_y, val_x):
        val_pred = []
        train_x =  np.log(train_x + config['setup']['log_eps'])
        val_x =  np.log(val_x + config['setup']['log_eps'])
        for chan in range(train_y.shape[1]):
            pr = PoissonRegressor(alpha=alpha, max_iter=500)
            pr.fit(train_x, train_y[:, chan])
            while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
                print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
                oldmax = pr.max_iter
                del pr
                pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
                pr.fit(train_x, train_y[:, chan])
            val_pred.append(pr.predict(val_x))
        val_rates_s = np.vstack(val_pred).T
        return np.clip(val_rates_s, 1e-9, 1e20)

    kf = KFold(n_splits=5)

    stds = [30,  40, 50, 60, ]
    # stds = [10, 20, 30,  40, 50, 60, 70, 80,  90,  110, 130, 150]
    
    filters = []
    for std in stds:
        kern_sd = int(round(std / 10))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filters.append(lambda x: np.convolve(x, window, 'same'))

    train_spikes_heldin = h5dict['all_train_spikes_heldin'][:, -1, :]
    train_spikes_heldout = h5dict['all_train_spikes_heldout'][:, -1, :]
    test_spikes_heldin = h5dict['test_spikes_heldin'][:, -1, :]
    test_spikes_heldout = h5dict['test_spikes_heldout'][:, -1, :]

    for std, filter in zip(stds, filters):
        for alpha in np.logspace(-3, 0, 4):
            split = []
            split_test = []
            for train_index, val_index in kf.split(train_spikes_heldin):
                heldin_smth_spikes= np.apply_along_axis(filter, 0, train_spikes_heldin)
                train_hi = heldin_smth_spikes[train_index]
                val_hi = heldin_smth_spikes[val_index]
                train_ho = train_spikes_heldout[train_index]
                val_ho = train_spikes_heldout[val_index]
                val_rates = fit_poisson(alpha, train_hi, train_ho, val_hi)
                split.append(bits_per_spike(np.expand_dims(val_rates, 1), np.expand_dims(val_ho, 1)))
            print('alpha:',alpha,'std:',std)
            print('val mean:', np.mean(np.array(split)))
            train_hi= np.apply_along_axis(filter, 0, train_spikes_heldin)
            train_ho = train_spikes_heldout
            test_hi= np.apply_along_axis(filter, 0, test_spikes_heldin)
            test_ho = test_spikes_heldout
            val_rates = fit_poisson(alpha, train_hi, train_ho, test_hi)
            print('test:', bits_per_spike(np.expand_dims(val_rates, 1), np.expand_dims(test_ho, 1)))


if __name__ == "__main__":
    get_lag_and_smth()






    # print('Generating data...')

    # name = model.name

    # if not os.path.isdir(f"plots"): os.makedirs(f"plots")
    # if not os.path.isdir(f"plots/{name}"): os.makedirs(f"plots/{name}")

    # # make_test_data(window=30, overlap=24, lag=lag, smooth_std=smth_std)
    # with h5py.File('/home/dmifsud/Projects/NDT-U/data/mc_rtt_cont_24_test.h5', 'r') as h5file:
    #     h5dict = h5_to_dict(h5file)

    # model.eval()

    # dataset = NWBDataset('/home/dmifsud/Projects/NDT-U/data/mc_rtt_train.nwb', split_heldout=True)

    # has_change = dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
    # change_nan = dataset.data[has_change].isna().any(axis=1)
    # drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
    # change_times = dataset.data.index[has_change]
    # start_times = change_times[:-1][~drop_trial]
    # end_times = change_times[1:][~drop_trial]
    # target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
    # reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
    # reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
    # dataset.trial_info = pd.DataFrame({
    #     'trial_id': np.arange(len(start_times)),
    #     'start_time': start_times,
    #     'end_time': end_times,
    #     'target_pos': target_pos,
    #     'reach_dist_x': reach_dist[:, 0],
    #     'reach_dist_y': reach_dist[:, 1],
    #     'reach_angle': reach_angle,
    # })

    # dataset.resample(10)

    # speed = np.linalg.norm(dataset.data.finger_vel, axis=1)
    # dataset.data['speed'] = speed
    # peak_times = dataset.calculate_onset('speed', 0.05)

    # dataset.smooth_spk(smth_std, name=f'smth_{smth_std}', ignore_nans=True)

    # lag_bins = int(round(lag / dataset.bin_width))
    # nans = dataset.data.finger_vel.x.isna().reset_index(drop=True)

    # vel = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].to_numpy()
    # vel_index = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].index

    # spikes_hi = dataset.data.spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()
    # spikes_ho = dataset.data.heldout_spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

    # print('Done!\n')

    # Need:
        # Test Rates
        # causal Smth Test Rates
        # acausal Smth Test Rates
        # causal Smth test spikes at 30, 50, & 80 ms
        # acausal Smth test spikes at 30, 50, & 80 ms
        # HE Test Rates
        # HE causal Smth Test Rates
        # HE acausal Smth Test Rates
        # ME Test Rates
        # ME causal Smth Test Rates
        # ME acausal Smth Test Rates
        # LE Test Rates
        # LE causal Smth Test Rates
        # LE acausal Smth Test Rates
        # Test Vel 
        # HE Vel 
        # ME Vel 
        # LE Vel 
        # .nwb file

    # Rates vs smoothed spikes (hi_all, ho_all, hi_indv, ho_indv) Test
        # Spike Smoothing dropdown (30ms, 50ms, 80ms)
        # Rate Smoothing types (none, causual, acausal)

    # plot_rates_vs_spks_all(wandb, 'heldin', )
    # plot_rates_vs_spks_indv(wandb, 'heldin')
    # plot_rates_vs_spks_all(wandb, 'heldout')
    # plot_rates_vs_spks_indv(wandb, 'heldout')

    # Test co-bps
        # Smoothing types (none, causual, acausal)
        # Exclusion types (HE, ME, LE)

     # Decoding Test trials
        # Exclusion types (All, HE, ME, LE)
        # Smoothing types (none, causual, acausal)

    #  PCA Plot
        # Split dropdown (Train, Test, All)
        # Smoothing dropdown (none, causual, acausal)
        # Exclusion dropdown (HE, ME, LE)

    # plot_pca()

    # Pred vs true movements Test
        # Exclusion dropdown (HE, ME, LE)
        # Rates Smoothing dropdown (none, causual, acausal)

    # plot_true_vs_pred_movements()













#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║                           TEST SET INFERENCE                           ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Running inference on the test set...')

#     with torch.no_grad():
#         train_rates = []
#         test_ho_spikes = []
#         for spikes, heldout_spikes in zip(
#             torch.Tensor(h5dict['test_spikes_heldin']).to(device), torch.Tensor(h5dict['test_spikes_heldout']).to(device)
#         ):
#             ho_spikes = torch.zeros_like(heldout_spikes).to(device)
#             spikes_new = torch.cat([spikes, ho_spikes], -1).to(device)
#             output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
#             train_rates.append(output.cpu())
#             test_ho_spikes.append(heldout_spikes.unsqueeze(dim=0)[:, -1, :].cpu())

#     train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates
#     test_ho_spikes = torch.cat(test_ho_spikes, dim=0) # turn into tensor and use exponential on rates

#     co_bps = float(bits_per_spike(train_rates[:, 98:].unsqueeze(dim=0).numpy(), test_ho_spikes.unsqueeze(dim=0).numpy()))
#     print(f'\n╔═══════════════════════════╗\n║ NDT Test Set Co-bps:      ║\n║   {co_bps:.3f}                   ║\n╚═══════════════════════════╝\n')
#     with open(f"plots/{name}/test_co_bps.txt", 'w') as f:
#         f.write(f'╔═══════════════════════════╗\n║ NDT Test Set Co-bps:      ║\n║   {co_bps:.3f}                   ║\n╚═══════════════════════════╝')
#     wandb.log({"Test Set Co-bps": co_bps})


#     gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv.fit(train_rates.numpy(), h5dict['test_vel_segments'])
#     print(f'╔═══════════════════════════╗\n║ NDT Test Set Decoding:    ║\n║   {gscv.best_score_:.3f} R\u00b2                ║\n╚═══════════════════════════╝')
#     with open(f"plots/{name}/velocity_decoding.txt", 'w') as f:
#         f.write(f'╔═══════════════════════════╗\n║ NDT Test Set Decoding:    ║\n║   {gscv.best_score_:.3f} R\u00b2                ║\n╚═══════════════════════════╝')
#     wandb.log({"Test Set Decoding": float(gscv.best_score_)})


#     smth_spikes = torch.Tensor(h5dict['test_hi_smth_spikes'])
#     heldout_smth_spikes = torch.Tensor(h5dict['test_ho_smth_spikes'])
#     smth_spikes = torch.cat([smth_spikes, heldout_smth_spikes], -1)

#     gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv.fit(smth_spikes.numpy(), h5dict['test_vel_segments'])
#     print(f'\n╔════════════════════════════════════╗\n║ Smoothed Spikes Test Set Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                         ║\n╚════════════════════════════════════╝\n')
#     with open(f"plots/{name}/velocity_decoding.txt", 'a') as f:
#         f.write(f'\n\n╔════════════════════════════════════╗\n║ Smoothed Spikes Test Set Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                         ║\n╚════════════════════════════════════╝')
#     wandb.log({"Smoothed Spikes Test Set Decoding": float(gscv.best_score_)})


#     print('Done!\n')


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║               HELDIN RATES VS SMTH SPIKES (RANGE SLIDER)               ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Generating "hi_indv_spk_vs_rates.html"...')

#     fig = go.Figure()
#     x_range=2500

#     fig.add_trace(go.Scatter(y=list(train_rates[:x_range,0]), line=dict(color="#e15759"), name="NDT Rates",))
#     fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,0]), line=dict(color="#4e79a7"), name="Smooth Spikes",))
#     for i in range(1, 98):
#         fig.add_trace(go.Scatter(y=list(train_rates[:x_range,i]), visible=False, line=dict(color="#e15759"), name="NDT Rates",))
#         fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,i]), visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(),
#             rangeslider=dict(visible=True)
#         )
#     )

#     buttons = []
#     for i in range(98):
#         vis_list = [False for i in range(196)]
#         vis_list[i*2] = True
#         vis_list[i*2+1] = True
#         buttons.append(dict(
#             method='restyle',
#             label=f'ch {i+1}',
#             visible=True,
#             args=[{'visible':vis_list,}]
#         ))
            
#     # specify updatemenu        
#     um = [{
#         'buttons':buttons, 
#         'direction': 'down',
#         'pad': {"r": 0, "t": 0, "b":20},
#         'showactive':True,
#         'x':0.5,
#         'xanchor':"center",
#         'y':1.00,
#         'yanchor':"bottom" 
#     }]
#     fig.update_layout(updatemenus=um)

#     fig['layout']['xaxis'].update(range=['0', '300'])

#     layout = go.Layout(
#         margin=go.layout.Margin(
#             l=60, #left margin
#             r=0, #right margin
#             b=0, #bottom margin
#             t=0  #top margin
#         )
#     )
#     fig.update_layout(layout)

#     fig.update_xaxes(
#         ticktext=[f'{int(i/100)}s' for i in range(0, x_range, 100)],
#         tickvals=[i for i in range(0, x_range, 100)],
#     )

#     fig.update_layout(
#         legend=dict(
#             yanchor="bottom",
#             y=1.035,
#             xanchor="right",
#             x=1.00,
#         ),
#         # xaxis_title="Time",
#         yaxis_title="Spikes per Window",
#         title="NDT Rates vs Smoothed Spikes - Heldin Channels",
#     )

#     config = {'displayModeBar': False}
#     fig.write_html(f"plots/{name}/hi_indv_spk_vs_rates.html", config=config)

#     print("Done!\n")


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║              HELDOUT RATES VS SMTH SPIKES (RANGE SLIDER)               ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Generating "ho_indv_spk_vs_rates.html"...')

#     fig = go.Figure()
#     x_range=2500

#     fig.add_trace(go.Scatter(y=list(train_rates[:x_range,98]), line=dict(color="#e15759"), name="NDT Rates",))
#     fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,98]), line=dict(color="#4e79a7"), name="Smooth Spikes",))
#     for i in range(99, 130):
#         fig.add_trace(go.Scatter(y=list(train_rates[:x_range,i]), visible=False, line=dict(color="#e15759"), name="NDT Rates",))
#         fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,i]), visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(),
#             rangeslider=dict(visible=True)
#         )
#     )

#     buttons = []
#     for i in range(98, 130):
#         vis_list = [False for i in range(64)]
#         vis_list[(i-98)*2] = True
#         vis_list[(i-98)*2+1] = True
#         buttons.append(dict(
#             method='restyle',
#             label=f'ch {i+1}',
#             visible=True,
#             args=[{'visible':vis_list,}]
#         ))
            
#     # specify updatemenu        
#     um = [{
#         'buttons':buttons, 
#         'direction': 'down',
#         'pad': {"r": 0, "t": 0, "b":20},
#         'showactive':True,
#         'x':0.5,
#         'xanchor':"center",
#         'y':1.00,
#         'yanchor':"bottom" 
#     }]
#     fig.update_layout(updatemenus=um)

#     fig['layout']['xaxis'].update(range=['0', '301'])

#     layout = go.Layout(
#         margin=go.layout.Margin(
#             l=60, #left margin
#             r=0, #right margin
#             b=0, #bottom margin
#             t=0  #top margin
#         )
#     )
#     fig.update_layout(layout)

#     fig.update_xaxes(
#         ticktext=[f'{int(i * 10)}ms' for i in range(0, x_range, 25)],
#         tickvals=[i for i in range(0, x_range, 25)],
#     )

#     fig.update_layout(
#         legend=dict(
#             yanchor="bottom",
#             y=1.035,
#             xanchor="right",
#             x=1.00,
#         ),
#         # xaxis_title="Time",
#         yaxis_title="Spikes per Second",
#         title="Rates vs Smoothed Spikes - Heldout Channels",
#     )

#     config = {'displayModeBar': False}
#     fig.write_html(f"plots/{name}/ho_indv_spk_vs_rates.html", config=config)

#     print("Done!\n")


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║               HELDIN RATES VS SMTH SPIKES (ALL CHANNELS)               ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Generating "hi_all_spk_vs_rates.html" & "hi_all_spk_vs_rates.js"...')

#     def rates_string(neuron):
#         array_string = 'y: ['
#         for i in train_rates[:1000,neuron]:
#             array_string += str(i.item())+','
#         array_string += '],'
#         return array_string

#     def ss_string(neuron):
#         array_string = 'y: ['
#         for i in smth_spikes[:1000,neuron]:
#             array_string += str(i.item())+','
#         array_string += '],'
#         return array_string

#     with open(f"plots/{name}/hi_all_spk_vs_rates.html", "w") as f:
#         f.write('<!DOCTYPE html><html lang="en" ><head><meta charset="UTF-8"><title>NDT Heldin Rates</title></head><body><!-- partial:index.partial.html --><div id="legend" style="height: 50px"></div><div style="height:450px; overflow-y: auto"><div id="plot" style="height:8000px"></div></div><div id="xaxis" style="height: 60px"></div><!-- partial --><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.3.1/plotly.min.js"></script><script  src="./hi_all_spk_vs_rates.js"></script></body></html>')

#     with open(f"plots/{name}/hi_all_spk_vs_rates.js", "w") as f:
#         names = []
#         for i in range(98):
#             names.append(f'trace{i+1}')
#             names.append(f'trace{i+1}r')
#             str_to_write = f'var trace{i+1} = {{'
#             str_to_write += ss_string(i)
#             str_to_write += f"marker: {{color: '#4e79a7'}},name: 'Smoothed Spikes',yaxis: 'y{i+1}',type: 'line',"
#             if i != 0:
#                 str_to_write += "showlegend: false,"
#             str_to_write += f'}};\nvar trace{i+1}r = {{'
#             str_to_write += rates_string(i)
#             str_to_write += f"marker: {{color: '#e15759'}},name: 'NDT Rates',yaxis: 'y{i+1}',type: 'line',"
#             if i != 0:
#                 str_to_write += "showlegend: false,"
#             str_to_write +='};\n'
#             f.write(str_to_write)
#         names_str = 'data = ['
#         for i in names:
#             names_str += f"{i}, "
#         names_str += ']'
#         f.write(names_str+f'\n')
#         f.write(f'let bottomTraces = [{{ mode: "scatter" }}];\nlet bottomLayout = {{yaxis: {{ tickmode: "array", tickvals: [], fixedrange: true }},xaxis: {{tickmode: "array",tickvals: [0, 25, 50, 75, 100],ticktext: ["0s", "2.5s", "5s", "7.5s", "10s"],range: [0, 100],domain: [0.0, 1.0],fixedrange: true}},margin: {{ l: 25, t: 0 , r: 40}},}};\nvar config = {{responsive: true, displayModeBar: false}};\nPlotly.react("plot",data,{{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: 98, columns: 1}},')
#         axis_labels = f"\nyaxis: {{title: {{text: 'ch 1',}}, showticklabels: false, fixedrange: true}},\n"
#         for i in range(2,99):
#             axis_labels += f"yaxis{i}: {{title: {{text: 'ch {i}',}}, showticklabels: false, fixedrange: true}},\n"
#         f.write(axis_labels)
#         f.write('margin: { l: 25, t: 25, b: 0 , r: 25},showlegend: false,},config);\nPlotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false, responsive: true });\ndata = [{y: [null],name: "Smooth Spikes",mode: "lines",marker: {color: "#4e79a7"},},{y: [null],name: "NDT Rates",mode: "lines",marker: {color: "#e15759"},}];\nlet newLayout = {title: {text:"Rates vs Smoothed Spikes - Heldin Channels", y:0.5, x:0.025},yaxis: { visible: false},xaxis: { visible: false},margin: { l: 0, t: 0, b: 0, r: 0 },showlegend: true,};\nPlotly.react("legend", data, newLayout, { displayModeBar: false, responsive: true });')

#     print("Done!\n")


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║              HELDOUT RATES VS SMTH SPIKES (ALL CHANNELS)               ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Generating "ho_all_spk_vs_rates.html" and "ho_all_spk_vs_rates.js"...')

#     def rates_string(neuron):
#         array_string = 'y: ['
#         for i in train_rates[:1000,neuron]:
#             array_string += str(i.item())+','
#         array_string += '],'
#         return array_string

#     def ss_string(neuron):
#         array_string = 'y: ['
#         for i in smth_spikes[:1000,neuron]:
#             array_string += str(i.item())+','
#         array_string += '],'
#         return array_string

#     with open(f"plots/{name}/ho_all_spk_vs_rates.html", "w") as f:
#         f.write('<!DOCTYPE html><html lang="en" ><head><meta charset="UTF-8"><title>NDT Heldout Rates</title></head><body><!-- partial:index.partial.html --><div id="legend" style="height: 50px"></div><div style="height:450px; overflow-y: auto"><div id="plot" style="height:2500px"></div></div><div id="xaxis" style="height: 60px"></div><!-- partial --><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.3.1/plotly.min.js"></script><script>\n')
#         # with open(f"plots/{name}/ho_all_spk_vs_rates.js", "w") as f:
#         names = []
#         for i in range(98, 130):
#             names.append(f'trace{i+1}')
#             names.append(f'trace{i+1}r')
#             str_to_write = f'var trace{i+1} = {{'
#             str_to_write += ss_string(i)
#             str_to_write += f"marker: {{color: '#4e79a7'}},name: 'Smoothed Spikes',yaxis: 'y{i-97}',type: 'line',"
#             if i != 0:
#                 str_to_write += "showlegend: false,"
#             str_to_write += f'}};\nvar trace{i+1}r = {{'
#             str_to_write += rates_string(i)
#             str_to_write += f"marker: {{color: '#e15759'}},name: 'NDT Rates',yaxis: 'y{i-97}',type: 'line',"
#             if i != 0:
#                 str_to_write += "showlegend: false,"
#             str_to_write +='};\n'
#             f.write(str_to_write)
#         names_str = 'data = ['
#         for i in names:
#             names_str += f"{i}, "
#         names_str += ']'
#         f.write(names_str+f'\n')
#         f.write(f'var config = {{responsive: true, displayModeBar: false}};')
        
#         f.write(f'var layout_hide_ticks = {{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: 32, columns: 1}},')
#         axis_labels = f"\nyaxis: {{title: {{text: 'ch 99',}}, showticklabels: false, fixedrange: true}},\n"
#         for i in range(100,131):
#             axis_labels += f"yaxis{i-98}: {{title: {{text: 'ch {i}',}}, showticklabels: false, fixedrange: true}},\n"
#         f.write(axis_labels)
#         f.write('margin: { l: 25, t: 45, b: 0 , r: 25},showlegend: false,}; ')

#         f.write(f'var layout_show_ticks = {{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: 32, columns: 1}},')
#         axis_labels = f"\nyaxis: {{title: {{text: 'ch 99',}}, showticklabels: true, fixedrange: true}},\n"
#         for i in range(100,131):
#             axis_labels += f"yaxis{i-98}: {{title: {{text: 'ch {i}',}}, showticklabels: true, fixedrange: true}},\n"
#         f.write(axis_labels)
#         f.write('margin: { l: 60, t: 45, b: 0 , r: 25},showlegend: false,}; ')

#         f.write('var updatemenus=[{buttons: [{args: [{}, {...layout_hide_ticks}], label: "Hide Y Axis Labels", method: "update"}, {args: [{}, {...layout_show_ticks}], label:"Show Y Axis Labels (spikes/sec)", method:"update"}], direction: "down", pad: {"r": 0, "t": 0, "b":0, "l": 0}, showactive: true, type: "dropdown", x: 0, xanchor: "left", y: 1.0175, yanchor: "top",}, {buttons: [{args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 1000]}}], label: "10s", method: "update"}, {args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 500]}}], label:"5s", method:"update"}, {args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 250]}}], ')
#         f.write('label:"2.5s", method:"update"}, {args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 100]}}], label:"1s", method:"update"}, {args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 50]}}], label:"500ms", method:"update"},], direction: "down", pad: {"r": 0, "t": 0, "b":0, "l": 0}, showactive: true, type: "dropdown", x: 0.5, xanchor: "center", y: 1.0175, yanchor: "top",},]; layout_hide_ticks["updatemenus"] = updatemenus; var config = {responsive: true, displayModeBar: false}; Plotly.react("plot",data,layout_hide_ticks,config);')
#         f.write('let bottomTraces = [{ mode: "scatter" }]; var graphDiv = document.getElementById("plot"); var axisDiv = document.getElementById("xaxis"); let bottomLayout = {yaxis: { tickmode: "array", tickvals: [], fixedrange: true },xaxis: {tickVals: [0, 250, 500, 750, 999], tickText: ["0s", "2.5s", "5s", "7.5s", "10s"], tickmode: "array",range: graphDiv.layout.xaxis.range,domain: [0.0, 1.0],fixedrange: true},margin: { l: 25, t: 0 , r: 40},}; Plotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false, responsive: true }); data = [{y: [null],name: "Smooth Spikes",mode: "lines",marker: {color: "#4e79a7"},},{y: [null],name: "NDT Rates",mode: "lines",marker: {color: "#e15759"},}];') 
#         f.write('let newLayout = {title: {text:"NDT Rates vs Smoothed Spikes - Heldout Channels (All)", y:0.5, x:0.025},yaxis: { visible: false},xaxis: { visible: false},margin: { l: 0, t: 0, b: 0, r: 0 },showlegend: true,}; Plotly.react("legend", data, newLayout, { displayModeBar: false, responsive: true }); var range = 1000; graphDiv.on("plotly_afterplot", function(){ var tickVals = [0, 250, 500, 750, 999]; var tickText = ["0s", "2.5s", "5s", "7.5s", "10s"]; if (graphDiv.layout.xaxis.range[1] == 1000) { var tickVals = [0, 250, 500, 750, 999]; var tickText = ["0s", "2.5s", "5s", "7.5s", "10s"]; range = 1000;} else if (graphDiv.layout.xaxis.range[1] == 500) {var tickVals = [0, 125, 250, 375, 500]; ')
#         f.write('var tickText = ["0s", "1.25s", "2.5s", "3.25s", "5s"]; range = 500;} else if (graphDiv.layout.xaxis.range[1] == 250) {var tickVals = [0, 62.5, 125, 187.5, 250]; var tickText = ["0s", "0.625s", "1.25s", "1.875s", "2.5s"]; range = 250;} else if (graphDiv.layout.xaxis.range[1] == 100) {var tickVals = [0, 25, 50, 75, 100]; var tickText = ["0s", "1.25s", "2.5s", "3.25s", "1s"]; range = 100;} else if (graphDiv.layout.xaxis.range[1] == 50) {var tickVals = [0, 12.5, 25, 37.5, 50]; var tickText = ["0ms", "125ms", "250ms", "325ms", "500ms"]; range = 50;};if (range != axisDiv.layout.xaxis.range[1]) {if (graphDiv.layout.yaxis.showticklabels) { Plotly.update(axisDiv, bottomTraces, {yaxis: { tickmode: "array", tickvals: [], fixedrange: true },xaxis: {tickmode: "array",tickvals: tickVals,ticktext: tickText,range: [0, range],domain: [0.0, 1.0],fixedrange: true},margin: { l: 60, t: 0 , r: 40},});} else { Plotly.update(axisDiv, bottomTraces, {yaxis: { tickmode: "array", tickvals: [], fixedrange: true },xaxis: {tickmode: "array",tickvals: tickVals,ticktext: tickText,range: [0, range],domain: [0.0, 1.0],fixedrange: true},margin: { l: 25, t: 0 , r: 40},});}} else if (graphDiv.layout.xaxis.range[1] == 999) { Plotly.update(graphDiv, bottomTraces, {xaxis: {visible: false, fixedrange: true, range: axisDiv.layout.xaxis.range}}); range = axisDiv.layout.xaxis.range[1]; if (graphDiv.layout.yaxis.showticklabels) {Plotly.update(axisDiv, bottomTraces, {yaxis: { tickmode: "array", tickvals: [], fixedrange: true },xaxis: {tickmode: "array",tickvals: tickVals,ticktext: tickText,range: [0, range],domain: [0.0, 1.0],fixedrange: true},margin: { l: 60, t: 0 , r: 40},});} else {Plotly.update(axisDiv, bottomTraces,')
#         f.write('{yaxis: { tickmode: "array", tickvals: [], fixedrange: true },xaxis: {tickmode: "array",tickvals: tickVals,ticktext: tickText,range: [0, range],domain: [0.0, 1.0],fixedrange: true},margin: { l: 25, t: 0 , r: 40},});}}});')
#         f.write('\n</script></body></html>')

#     print("Done!\n")


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║                      RATES VELOCITY DECODING R^2                       ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Training an OLE on NDT Rates...')

#     def chop_and_infer(func,
#                     data,
#                     seq_len=30,
#                     stride=1,
#                     batch_size=64,
#                     output_dim=None,
#                     func_kw={}
#     ):
#         device = torch.device('cuda:0')
#         data_len, data_dim = data.shape[0], data.shape[1]
#         output_dim = data_dim if output_dim is None else output_dim

#         batch = np.zeros((batch_size, seq_len, data_dim), dtype=np.float64)
#         output = np.zeros((data_len, output_dim), dtype=np.float64)
#         olap = seq_len - stride

#         n_seqs = (data_len - seq_len) // stride + 1
#         n_batches = np.ceil(n_seqs / batch_size).astype(int)

#         i_seq = 0  # index of the current sequence
#         for i_batch in range(n_batches):
#             n_seqs_batch = 0  # number of sequences in this batch
#             start_ind_batch = i_seq * stride
#             for i_seq_in_batch in range(batch_size):
#                 if i_seq < n_seqs:
#                     start_ind = i_seq * stride
#                     batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
#                                                     seq_len]
#                     i_seq += 1
#                     n_seqs_batch += 1
#             end_ind_batch = start_ind + seq_len
#             batch_out = func(torch.Tensor(batch).to(device), **func_kw)[:n_seqs_batch]
#             n_samples = n_seqs_batch * stride
#             if start_ind_batch == 0:  # fill in the start of the sequence
#                 output[:olap, :] = batch_out[0, :olap, :].detach().cpu().numpy()
#             out_idx_start = start_ind_batch + olap
#             out_idx_end = end_ind_batch
#             out_slice = np.s_[out_idx_start:out_idx_end]
#             output[out_slice, :] = batch_out[:, olap:, :].reshape(
#                 n_samples, output_dim).detach().cpu().numpy()

#         return output

#     with torch.no_grad():
#         spikes = torch.Tensor(spikes_hi)
#         heldout_spikes = torch.Tensor(spikes_ho)
#         ho_spikes = torch.zeros_like(heldout_spikes)
#         spikes_new = torch.cat([spikes, ho_spikes], -1)
#         output = chop_and_infer(
#             model, 
#             spikes_new.numpy(),
#             seq_len=30,
#             stride=1
#         )
#     rates = np.exp(output)

#     gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv.fit(rates, vel)
#     print(f'\n╔══════════════════════════════╗\n║ NDT Filered Trials Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                   ║\n╚══════════════════════════════╝')
#     with open(f"plots/{name}/velocity_decoding.txt", 'a') as f:
#         f.write(f'\n\n╔══════════════════════════════╗\n║ NDT Filered Trials Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                   ║\n╚══════════════════════════════╝')
#     wandb.log({"Filered Trials Decoding": float(gscv.best_score_)})

#     pred_vel = gscv.predict(rates)

#     pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
#     dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

#     for i in range(rates.shape[1]):
#         mean = rates[:,i].mean()
#         std = rates[:,i].std()
#         rates[:,i] -= mean
#         rates[:,i] /= std

#     pca = PCA(n_components=3)
#     pca.fit(rates)
#     pca_comps = pca.transform(rates)

#     pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('pca', 'x'), ('pca', 'y'), ('pca', 'z')]))
#     dataset.data = pd.concat([dataset.data, pca_df], axis=1)


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║                    SMTH SPIKES VELOCITY DECODING R^2                   ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('\nTraining an OLE on Smoothed Spikes...')

#     rates = dataset.data.spikes_smth_50[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

#     gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
#     gscv.fit(rates, vel)
#     print(f'\n╔═══════════════════════════════════════════╗\n║ Smoothed Spikes Filtered Trials Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                                ║\n╚═══════════════════════════════════════════╝\n')
#     with open(f"plots/{name}/velocity_decoding.txt", 'a') as f:
#         f.write(f'\n\n╔═══════════════════════════════════════════╗\n║ Smoothed Spikes Filtered Trials Decoding: ║\n║   {gscv.best_score_:.3f} R\u00b2                                ║\n╚═══════════════════════════════════════════╝')
#     wandb.log({"Smoothed Spikes Filtered Trials Decoding": float(gscv.best_score_)})

#     smth_pred_vel = gscv.predict(rates)

#     smth_pred_vel_df = pd.DataFrame(smth_pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pred_vel', 'x'), ('smth_pred_vel', 'y')]))
#     dataset.data = pd.concat([dataset.data, smth_pred_vel_df], axis=1)

#     for i in range(rates.shape[1]):
#         mean = rates[:,i].mean()
#         std = rates[:,i].std()
#         rates[:,i] -= mean
#         rates[:,i] /= std

#     pca = PCA(n_components=3)
#     pca.fit(rates)
#     pca_comps = pca.transform(rates)

#     pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pca', 'x'), ('smth_pca', 'y'), ('smth_pca', 'z')]))
#     dataset.data = pd.concat([dataset.data, pca_df], axis=1)


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║                   PREDICTED MOVEMENT (TRIAL SLIDER)                    ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('\nGenerating "true_vs_pred_movement.html"...')

#     trial_data = dataset.make_trial_data(align_field='speed_onset', align_range=(-290, 750), allow_nans=True)

#     fig = go.Figure()

#     for tid, trial in trial_data.groupby('trial_id'):
#         x = np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)
#         y = trial.pred_vel.to_numpy()[29:, 1]/100
#         fig.add_trace(go.Scatter(visible=False, line=dict(color="#e15759"), x=np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100), name="NDT Predicted Reach"))
#         fig.add_trace(go.Scatter(visible=False, line=dict(color="#4e79a7"), x=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100), name="Smooth Spikes Predicted Reach"))
#         fig.add_trace(go.Scatter(visible=False, line=dict(color="#000000"), x=np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100), name="True Reach"))

#     ranges = []
#     for tid, trial in trial_data.groupby('trial_id'):
#         min_x = min(min(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)), min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100)), min(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100)))
#         min_y = min(min(np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100)), min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100)), min(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100)))
#         max_x = max(max(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)), max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100)), max(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100)))
#         max_y = max(max(np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100)), max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100)), max(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100)))
#         pad = 0.05

#         x_len = (max_x - min_x) * pad
#         y_len = (max_y - min_y) * pad

#         ranges.append(([ min_x - x_len, max_x + x_len], [min_y - y_len, max_y + y_len]))

#     fig.data[0].visible = True
#     fig.data[1].visible = True
#     fig.data[2].visible = True

#     steps = []
#     for i in range(int(len(fig.data)/3)):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)},
#                 #   {"title": "Slider switched to step: " + str(i), 
#                 {"xaxis" : dict(
#                     range=ranges[i][0], 
#                     tickmode = 'linear',
#                     tick0=0,
#                     dtick=10, 
#                     zeroline=True, 
#                     zerolinewidth=2, 
#                     zerolinecolor='slategray',
#                     title="Horizontal Movement Distance (mm)", 
#                     fixedrange=True 
#                 ),
#                 "yaxis" : dict(
#                     scaleanchor = "x", 
#                     scaleratio = 1, 
#                     range=ranges[i][1], 
#                     zeroline=True, 
#                     zerolinewidth=2, 
#                     zerolinecolor='slategray',
#                     tickmode = 'linear',
#                     tick0=0,
#                     dtick=10,
#                     title="Vertical Movement Distance (mm)", 
#                     fixedrange=True 
#                 )}],
#             label=f'{i}'
#         )
#         step["args"][0]["visible"][i*3] = True  # Toggle i'th trace to "visible"
#         step["args"][0]["visible"][i*3+1] = True  # Toggle i'th trace to "visible"
#         step["args"][0]["visible"][i*3+2] = True  # Toggle i'th trace to "visible"
#         steps.append(step)

#     sliders = [dict(
#         active=0,
#         currentvalue={"prefix": "Trial: "},
#         pad={"t": 50},
#         steps=steps
#     )]

#     fig.update_layout(
#         sliders=sliders,
#         legend=dict(
#             yanchor="bottom",
#             y=1.035,
#             xanchor="right",
#             x=1.00
#         ),
#         xaxis_title="Horizontal Movement Distance (mm)",
#         yaxis_title="Vertical Movement Distance (mm)",
#         title="True vs Predicted Movements",
#     )

#     fig.update_xaxes(
#         range=ranges[0][0], 
#         tickmode = 'linear',
#         tick0=0,
#         dtick=10, 
#         zeroline=True, 
#         zerolinewidth=2, 
#         zerolinecolor='slategray', 
#         fixedrange=True
#     )
#     fig.update_yaxes(
#         scaleanchor = "x", 
#         scaleratio = 1, 
#         range=ranges[0][1], 
#         zeroline=True, 
#         zerolinewidth=2, 
#         zerolinecolor='slategray',
#         tickmode = 'linear',
#         tick0=0,
#         dtick=10, 
#         fixedrange=True
#     )
#     layout = go.Layout(
#         margin=go.layout.Margin(
#             l=65, #left margin
#             r=25, #right margin
#             b=135, #bottom margin
#             t=0  #top margin
#         )
#     )
#     fig.update_layout(layout)
#     config = {'displayModeBar': False}
#     fig.write_html(f"plots/{name}/true_vs_pred_movement.html", config=config)
#     wandb.log({"pred movement plot": wandb.Html(open(f"plots/{name}/true_vs_pred_movement.html"), inject=False)})


#     print("Done!\n")


#     '''
#     ╔════════════════════════════════════════════════════════════════════════╗
#     ║                             RATES PCA PLOT                             ║
#     ╚════════════════════════════════════════════════════════════════════════╝
#     '''
#     print('Generating "rates_pca.html"...')

#     norm = colors.Normalize(vmin=-180, vmax=180, clip=True)
#     mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
#     mapper.set_array([])

#     fig = go.Figure()

#     for tid, trial in trial_data.groupby('trial_id'):
#         angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#         fig.add_trace(
#             go.Scatter3d(
#                 x=trial.pca.x, 
#                 y=trial.pca.y, 
#                 z=trial.pca.z,
#                 mode='lines',
#                 line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(angle))}')
#             )
#         )

#     fig.update_layout(
#         width=500,
#         height=500,
#         autosize=False,
#         showlegend=False,
#         # title="PCA of NDT Rates",
#         scene=dict(
#             xaxis_showspikes=False,
#             yaxis_showspikes=False,
#             zaxis_showspikes=False,
#             xaxis_title="PC1",
#             yaxis_title="PC2",
#             zaxis_title="PC3",
#             camera=dict(
#                 center=dict(
#                     x=0.0,
#                     y=0.0,
#                     z=-0.125,
#                 ),
#             ),
#             aspectratio = dict( x=1, y=1, z=1 ),
#             aspectmode = 'manual'
#         ),
#     )

#     # fig.add_layout_image(
#     #     dict(
#     #         source="https://domenick-m.github.io/NDT-Timing-Test/plots/color_wheel.png",
#     #         xref="paper", yref="paper",
#     #         x=1.085, y=0.01,
#     #         sizex=0.35, sizey=0.35,
#     #         xanchor="right", yanchor="bottom"
#     #     )
#     # )

#     fig.update_layout(margin=dict(r=0, l=0, b=0, t=0))

#     config = {'displayModeBar': False}
#     fig.write_html(f"plots/{name}/rates_pca.html", config=config)

#     # for line in fileinput.input(f"plots/{name}/rates_pca.html", inplace=True):
#     #     if line == "    <div>                        <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n":
#     #         print("    <div align=\"center\">                        <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n", end='')
#     #     else:
#     #         print(line, end='')
#     wandb.log({"pca plot": wandb.Html(open(f"plots/{name}/rates_pca.html"), inject=False)})


#     print("Done!\n")


#     '''
#     # ╔════════════════════════════════════════════════════════════════════════╗
#     # ║                          SMTH SPIKES PCA PLOT                          ║
#     # ╚════════════════════════════════════════════════════════════════════════╝
#     # '''
#     print('Generating "smth_pca.html"...')

#     fig = go.Figure()

#     for tid, trial in trial_data.groupby('trial_id'):
#         angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#         fig.add_trace(
#             go.Scatter3d(
#                 x=trial.smth_pca.x, 
#                 y=trial.smth_pca.y, 
#                 z=trial.smth_pca.z,
#                 mode='lines',
#                 line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(angle))}')
#             )
#         )

#     fig.update_layout(
#         width=300,
#         height=300,
#         autosize=False,
#         showlegend=False,
#         # title="PCA of Smoothed Spikes",
#         scene=dict(
#             xaxis_showspikes=False,
#             yaxis_showspikes=False,
#             zaxis_showspikes=False,
#             xaxis_title="PC1",
#             yaxis_title="PC2",
#             zaxis_title="PC3",
#             camera=dict(
#                 center=dict(
#                     x=0.0,
#                     y=0.0,
#                     z=-0.125,
#                 ),
#             ),
#             aspectratio = dict( x=1, y=1, z=1 ),
#             aspectmode = 'manual'
#         ),
#     )

#     # fig.add_layout_image(
#     #     dict(
#     #         source="https://domenick-m.github.io/NDT-Timing-Test/plots/color_wheel.png",
#     #         xref="paper", yref="paper",
#     #         x=1.085, y=0.01,
#     #         sizex=0.35, sizey=0.35,
#     #         xanchor="right", yanchor="bottom"
#     #     )
#     # )

#     fig.update_layout(margin=dict(r=0, l=0, b=10, t=0))

#     config = {'displayModeBar': False}
#     fig.write_html(f"plots/{name}/smth_pca.html", config=config)

#     # for line in fileinput.input(f"plots/{name}/smth_pca.html", inplace=True):
#     #     if line == "    <div>                        <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n":
#     #         print("    <div align=\"center\">                        <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n", end='')
#     #     else:
#     #         print(line, end='')

#     print("Done!\n")


# # END

