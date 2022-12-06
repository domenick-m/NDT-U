
import sys
sys.path.append('../')
from utils.multisession_utils import align_sessions
from utils.config_utils import get_config
from utils.plot.pcs import plot_pcs
import numpy as np
import wandb
import os.path as osp
from utils.toolkit_utils import load_toolkit_datasets, get_trialized_data
config = get_config()
# wandb.init(project='plots', name='Alignment Move Onset Testing')
datasets = load_toolkit_datasets(config)

for i in range(2, 21):
    config.defrost()
    config.data.cl_align_field = 'start_time'
    config.data.cl_align_range = [0, i*100]
    config.freeze()
    so_alignment_matrices, so_alignment_biases = align_sessions(config)
    # config.defrost()
    # config.data.ol_align_field = 'speed_onset'
    # config.data.ol_align_range = [-350, 1250]

    # config.data.cl_align_field = 'start_time'
    # config.data.cl_align_range = [0, 1000]
    # config.freeze()
    trialized_data = get_trialized_data(config, datasets)
    ol_cond_avg = ([], []) 
    cl_cond_avg = ([], []) 
    ol_single_trial = ([], [])
    cl_single_trial = ([], [])

    trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size
    cl_trial_len = (config.data.cl_align_range[1] - config.data.cl_align_range[0]) / config.data.bin_size

    for idx, session in enumerate(config.data.sessions):    
        # for cond_id, trials in trialized_data[session]['ol_trial_data'].groupby(('cond_id', 'n')):
        #     if cond_id != 0:
        #     # if cond_id == 1:
        #         low_d_trials = []
        #         for trial_id, trial in trials.groupby('trial_id'):
        #             heldin_spikes = trial.spikes_smth.to_numpy()[:, datasets[session].heldin_channels]
        #             if heldin_spikes.shape[0] == trial_len:
        #                 low_d_trial = np.dot(heldin_spikes, so_alignment_matrices[idx].T)
        #                 low_d_trial = low_d_trial + np.array(so_alignment_biases[idx])
        #                 low_d_trials.append(low_d_trial)

        #         ol_single_trial[0].append(np.concatenate(low_d_trials, 0))
        #         ol_single_trial[1].append(cond_id)

        #         ol_cond_avg[0].append(np.array(low_d_trials).mean(0))
        #         ol_cond_avg[1].append(cond_id)

        for cond_id, trials in trialized_data[session]['cl_trial_data'].groupby(('cond_id', 'n')):
            if cond_id != 0:
                low_d_trials = []
                for trial_id, trial in trials.groupby('trial_id'):
                    heldin_spikes = trial.spikes_smth.to_numpy()[:, datasets[session].heldin_channels]
                    if heldin_spikes.shape[0] == cl_trial_len:
                        low_d_trial = np.dot(heldin_spikes, so_alignment_matrices[idx].T)
                        low_d_trial = low_d_trial + np.array(so_alignment_biases[idx])
                        low_d_trials.append(low_d_trial)

                cl_single_trial[0].append(np.concatenate(low_d_trials, 0))
                cl_single_trial[1].append(cond_id)

                cl_cond_avg[0].append(np.array(low_d_trials).mean(0))
                cl_cond_avg[1].append(cond_id)
    # fig = plot_pcs(*ol_cond_avg, 'OL Condition Averaged', return_fig=True)
    # fig.show()
    # fig = plot_pcs(*ol_single_trial, 'OL Single Trial', return_fig=True)
    # fig.show()
    fig = plot_pcs(*cl_single_trial, 'CL Single Trial', return_fig=True)
    html_str = fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')

    # with open(f'sing_plot_{i*100}.html', 'w') as f: 
    #     f.write(html_str)

    fig = plot_pcs(*cl_cond_avg, 'OL Condition Averaged', return_fig=True)
    html_str = fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')

    with open(f'low_avg_plot_{i*100}.html', 'w') as f: 
        f.write(html_str)