# %%
import copy
import math
import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import rds
from alignment_analysis.common import utils
from alignment_analysis.run_scripts.nomad_interface import NomadInterface
from alignment_analysis.run_scripts import nomad_plotting as nplt
from lfads_wrapper.run_posterior_mean_sampling import get_hps
from run_lfadslite import hps_dict_to_obj

plt.style.use('tableau-colorblind10')

# %% Define parameters
monkey = 'Jango'
datasets = [
    'Jango_20150730_001',
    'Jango_20150731_001',
    # 'Jango_20150805_001',
    # 'Jango_20150826_001',
    # 'Jango_20151102_001'
]
fmod = 'DS18(Jango_2015)/xds/'
binsize = 0.020  # s
smooth_ms = 60

channel_cutoff = 0.2
fixed_channel_count = 100

remove_high_corr = True if channel_cutoff else False
kfield = 'kin_v'
# %% Get the data
cond_datas = []
model_nis = []
for dataset in datasets:
    model_ni = NomadInterface(monkey, fmod, dataset)
    model_ni.load_dataset(binsize=binsize,
                          remove_high_corr=remove_high_corr,
                          sort_channels=True,
                          fixed_channel_count=fixed_channel_count,
                          show_plot=False,
                          cutoff=channel_cutoff)

    # Condition-average
    # smooth
    ss_name = rds.analyzer.smooth_spikes(model_ni.dataset, smooth_ms)

    # get successful trials
    model_ni.dataset.make_trials()
    selection = {'result': 'R'}
    model_ni.dataset.select_trials('successful', selection)

    # align to movement onset
    calculate_params = {
        'name': 'moveOnset',
        'threshold': 0.15,
    }

    margins = 0.0
    align_params = {'point': 'backward_move_onset', 'window': (0.25, 0.5)}
    extract_params = {
        'calculate_params': calculate_params,
        'align_params': align_params,
        'selection': selection,
        'margins': margins
    }

    extract_fields = list(model_ni.dataset.data.columns.levels[0])
    cond_sep_field = 'tgtDir'
    cond_data = rds.analyzer.get_aligned_data_fields(
        model_ni.dataset,
        extract_fields,
        extract_params=extract_params,
        cond_sep_field=cond_sep_field)

    # convert entries to lists because tuples are immutable
    for field in cond_data:
        cond_data[field] = list(cond_data[field])

    cond_datas.append(copy.deepcopy(cond_data))
    model_nis.append(copy.deepcopy(model_ni))

# %% global PCA
data_Xs = []
ndatasets = len(cond_datas)


def get_global_pcs(cond_datas, nfield, dataset_inds=None):
    if dataset_inds is None:
        dataset_inds = list(range(len(cond_datas)))

    for iDS in dataset_inds:
        data = cond_datas[iDS][nfield][0].transpose((1, 2, 0))
        data_Xs.append(copy.deepcopy(data))

    all_data = np.stack(tuple(data_Xs), axis=3)
    nchans, nconds, nbins, ndatasets = all_data.shape
    # change to nchans x ndatasets x nconds x nbins
    all_data = all_data.transpose((0, 3, 1, 2))
    all_data = all_data.reshape((nchans * ndatasets, nconds * nbins))

    # mean-center
    all_data_means = all_data.mean(axis=1)
    all_data_centered = (all_data.T - all_data_means.T).T

    pca = PCA(n_components=int(nchans / 50))

    pca.fit(all_data_centered.T)
    dim_reduced_data = np.dot(all_data_centered.T, pca.components_.T).T

    all_data = all_data.reshape((nchans, ndatasets, nconds, nbins))
    return dim_reduced_data, all_data, all_data_means


# %% Plot global low-D projections
dim_reduced_data, all_data, all_data_means = get_global_pcs(cond_datas,
                                                            nfield=ss_name,
                                                            dataset_inds=None)
nchans, ndatasets, nconds, nbins = all_data.shape
nPCs = dim_reduced_data.shape[0]

nrows, ncols = (2, 2)
fig, axes = plt.subplots(nrows=nrows,
                         ncols=ncols,
                         figsize=(ncols * 3 * 2, nrows * 2 * 2))
iPC = 0
plt_data = dim_reduced_data.reshape((nPCs, nconds, nbins))
for irow in range(nrows):
    for icol in range(ncols):
        ax = axes[irow, icol]
        ax.plot(plt_data[iPC, :, :].T)
        ax.set_title('PC ' + str(iPC))
        # ax.legend(list(range(nconds)))
        iPC += 1
# %% PCR
nfield = ss_name
nfield_aligned = nfield + '_aligned'
two_day_alignment = False


def run_pcr(cond_datas, dim_reduced_data, nfield, two_day_alignment=False):
    dim_reduced_data_means = dim_reduced_data.mean(axis=1)
    dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T).T

    nfield_aligned = nfield + '_aligned'

    alignment_matrices = []
    alignment_biases = []
    for iDS in range(ndatasets):
        if two_day_alignment:
            dataset_inds = [0, iDS]
            reload_data = False
        else:
            dataset_inds = None
            reload_data = False

        if reload_data or iDS == 0:
            dim_reduced_data, all_data, all_data_means = get_global_pcs(
                cond_datas, nfield=ss_name, dataset_inds=dataset_inds)

        cond_datas[iDS][nfield_aligned] = [None] * 2
        cond_datas[iDS][nfield_aligned][1] = [None] * nconds
        # get data
        this_dataset_data = all_data.reshape(
            (nchans, ndatasets, nconds * nbins))[:, iDS, :].squeeze()
        # mean-center
        this_dataset_means = np.squeeze(
            all_data_means.reshape(nchans, ndatasets)[:, iDS])
        this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

        # alignment
        reg = Ridge(alpha=1.0, fit_intercept=False)
        reg.fit(this_dataset_centered.T, dim_reduced_data_this.T)

        alignment_matrices.append(np.copy(reg.coef_.T))  # nchans x nPCs
        alignment_biases.append(copy.deepcopy(this_dataset_means))

        # align condition-averaged data
        cond_avg_data = np.dot(alignment_matrices[iDS].T,
                               this_dataset_centered)
        cond_avg_data = cond_avg_data.reshape((nPCs, nconds, nbins))

        # add to cond_datas with shape (nbins, nPCs, nconds)
        cond_datas[iDS][nfield_aligned][0] = cond_avg_data.transpose((2, 0, 1))

        # get single-trial low-D projections
        for icond in range(nconds):
            ntrials = cond_datas[iDS][nfield][1][icond].shape[2]
            this_dataset_centered = np.zeros(
                cond_datas[iDS][nfield][1][icond].shape, dtype=float)
            single_trial_data = np.zeros((nPCs, nbins, ntrials), dtype=float)
            for itrial in range(ntrials):
                this_dataset_centered[:, :, itrial] = (
                    cond_datas[iDS][nfield][1][icond][:, :, itrial].squeeze() -
                    this_dataset_means.T)
                single_trial_data[:, :, itrial] = np.dot(
                    alignment_matrices[iDS].T,
                    this_dataset_centered[:, :, itrial].squeeze().T)
            cond_datas[iDS][nfield_aligned][1][icond] = (
                single_trial_data.transpose((1, 0, 2)))

    return cond_datas, alignment_matrices, alignment_biases


cond_datas, alignment_matrices, alignment_biases = run_pcr(
    cond_datas, dim_reduced_data, nfield, two_day_alignment=False)
# %% plot channel PSTHs
for iDS in range(ndatasets):
    # get data
    this_dataset_data = all_data.reshape(
        (nchans, ndatasets, nconds * nbins))[:, iDS, :].squeeze()
    # mean-center
    this_dataset_means = np.squeeze(
        all_data_means.reshape(nchans, ndatasets)[:, iDS])
    this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T
    # remake condition axis
    this_dataset_centered_cond = this_dataset_centered.reshape(
        nchans, nconds, nbins)

    nrows, ncols = (4, 4)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(ncols * 3 * 2, nrows * 2 * 2))
    ichan = 0
    for irow in range(nrows):
        for icol in range(ncols):
            ax = axes[irow, icol]
            while np.sum(this_dataset_centered_cond[ichan, :, :]) == 0:
                ichan += 1
            ax.plot(this_dataset_centered_cond[ichan, :, :].T)
            ax.set_title('Ch. ' + str(ichan))
            ichan += 1
    plt.suptitle(datasets[iDS])
    plt.savefig('chans_{}.png'.format(datasets[iDS]))

# %% plot condition-averaged low-D
for iDS in range(ndatasets):
    cond_avg_data = cond_datas[iDS][nfield_aligned][0].transpose((1, 2, 0))
    nrows, ncols = (2, 2)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(ncols * 6, nrows * 4))
    iPC = 0
    for irow in range(nrows):
        for icol in range(ncols):
            ax = axes[irow, icol]
            ax.plot(cond_avg_data[iPC, :, :].T)
            ax.set_title('PC ' + str(iPC))
            iPC += 1
    plt.suptitle(datasets[iDS])
    plt.savefig('pcs_{}.png'.format(datasets[iDS]))

# %% plot single-trial low-D projections
icond = 0
for iDS in range(ndatasets):
    single_trial_data = cond_datas[iDS][nfield_aligned][1][icond].transpose(
        (1, 0, 2))
    nrows, ncols = (2, 2)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(ncols * 6, nrows * 4))
    iPC = 0
    for irow in range(nrows):
        for icol in range(ncols):
            ax = axes[irow, icol]
            ax.plot(single_trial_data[iPC, :, :], color=cm.tab10(icond))
            ax.set_title('PC ' + str(iPC))
            iPC += 1
    plt.suptitle(datasets[iDS])
    plt.savefig('pcs_single_trial_{}.png'.format(datasets[iDS]))

# %% plot condition-averaged single-trial low-D projections
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for iDS in range(ndatasets):
    nrows, ncols = (2, 2)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(ncols * 6, nrows * 4))
    iPC = 0
    for irow in range(nrows):
        for icol in range(ncols):
            ax = axes[irow, icol]
            for icond in range(nconds):
                single_trial_data = cond_datas[iDS][nfield_aligned][1][
                    icond].transpose((1, 0, 2))
                ax.plot(np.mean(single_trial_data[iPC, :, :], axis=1),
                        color=colors[icond])
            ax.set_title('PC ' + str(iPC))
            iPC += 1
    plt.suptitle(datasets[iDS])
    plt.savefig('pcs_cond_avg_{}.png'.format(datasets[iDS]))

# %% plot single-trial low-D projections grid
nrows, ncols = (4, ndatasets)
fig, axes = plt.subplots(nrows=nrows,
                         ncols=ncols,
                         sharey='row',
                         sharex=True,
                         figsize=(ncols * 6, nrows * 4))
icond = 0
for iDS in range(ndatasets):
    icol = iDS
    single_trial_data = cond_datas[iDS][nfield_aligned][1][icond].transpose(
        (1, 0, 2))
    iPC = 0
    for irow in range(nrows):
        ax = axes[irow, icol]
        ax.plot(single_trial_data[iPC, :, :], color=cm.tab10(icond))
        if irow == 0:
            ax.set_title(datasets[iDS] + '\n\nPC ' + str(iPC))
        else:
            ax.set_title('PC ' + str(iPC))
        iPC += 1
    plt.savefig('pcs_single_trial_grid.png')

# %% plot condition-averaged low-D projections grid
nrows, ncols = (4, ndatasets)
fig, axes = plt.subplots(nrows=nrows,
                         ncols=ncols,
                         sharey='row',
                         sharex=True,
                         figsize=(ncols * 6, nrows * 4))
icond = 0
for iDS in range(ndatasets):
    icol = iDS
    cond_avg_data = cond_datas[iDS][nfield_aligned][0].transpose((1, 2, 0))
    iPC = 0
    for irow in range(nrows):
        ax = axes[irow, icol]
        ax.plot(cond_avg_data[iPC, :, :].T)
        if irow == 0:
            ax.set_title(datasets[iDS] + '\n\nPC ' + str(iPC))
        else:
            ax.set_title('PC ' + str(iPC))
        iPC += 1
    plt.savefig('pcs_grid.png')

# %% Decoding
d_results = [None] * ndatasets
neural_fields = [ss_name, nfield_aligned]
decode_to = ['kin_v']
n_history = 3
for iDS in range(ndatasets):
    d_results[iDS] = utils.decode_all(model_nis[iDS],
                                      cond_datas[iDS],
                                      neural_fields,
                                      decode_to=decode_to,
                                      save=False,
                                      load=False,
                                      subpath=None,
                                      n_history=n_history,
                                      metric='vaf')

# %% Predict from day 0 decoder
use_cond_avg_data = False
for iDS in range(ndatasets):
    for nfield in neural_fields:
        for kfield in d_results[iDS][nfield]:
            if not use_cond_avg_data:
                nfield_data = cond_datas[iDS][nfield][1]
                kfield_data = cond_datas[iDS][kfield][1]
            else:
                # predict from cond-averaged data
                nfield_data = [None] * nconds
                kfield_data = [None] * nconds
                for icond in range(nconds):
                    nfield_data[icond] = np.mean(
                        cond_datas[iDS][nfield][1][icond], axis=2)[:, :,
                                                                   np.newaxis]
                    kfield_data[icond] = np.mean(
                        cond_datas[iDS][kfield][1][icond], axis=2)[:, :,
                                                                   np.newaxis]

            vaf, wf, pred = nplt.WF_decode_predict(
                d_results[0][nfield][kfield]['W'],
                nfield_data,
                kfield_data,
                n_history=n_history,
                metric='vaf')

            d_results[iDS][nfield][kfield]['day0_W'] = d_results[0][nfield][
                kfield]['W']
            d_results[iDS][nfield][kfield]['day0_vaf'] = vaf
            d_results[iDS][nfield][kfield]['day0_pred'] = pred

# %% Decoding results
for iDS in range(ndatasets):
    figure_path = os.getcwd()
    for nfield in d_results[iDS]:
        for kfield in d_results[iDS][nfield]:
            # decoding plots
            if kfield == 'wrist_raw_emg' or kfield == 'wrist_emg':
                ylabels = cond_data[kfield][2]
            else:
                ylabels = ['x', 'y']

            save_name = kfield + '_true'
            fig, ax = nplt.plot_condition_grid(
                d_results[iDS][nfield][kfield]['wf'],
                extract_params['align_params'],
                label='True',
                to_save=False,
                save_path=figure_path,
                save_name=save_name,
                ylabels=ylabels)
            title = (datasets[iDS] + ' ' + kfield + ' decoding from ' +
                     nfield + ', VAF: ' +
                     str(d_results[iDS][nfield][kfield]['day0_vaf']))
            save_name = '{}_{}_from_{}'.format(datasets[iDS], kfield, nfield)
            fig, ax = nplt.plot_condition_grid(
                d_results[iDS][nfield][kfield]['day0_pred'],
                extract_params['align_params'],
                fig_ax=[fig, ax],
                color='r',
                title=title,
                legend=True,
                label='Decoded',
                to_save=True,
                save_path=figure_path,
                save_name=save_name,
                ylabels=ylabels)

# %%