import sys
from utils_f import (get_config, parse_args)
from data.t5_dataset import T5CursorDataset
import pandas as pd
import numpy as np
import copy
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import plotly.graph_objects as go


arg_dict = parse_args(sys.argv[1:])
config = get_config(arg_dict)

datasets = {}

for session in [*config.data.pretrain_sessions, *config.data.finetune_sessions]:
    if not session in datasets:
        datasets[session] = T5CursorDataset(f'{config.data.dir}/{session}.mat')

session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

avg_conds = {}
trials_dict = {}

cond_list = None

for session in config.data.pretrain_sessions:
    dataset = copy.deepcopy(datasets[session])
    dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)
    dataset.resample(config.data.bin_size / 1000)
    dataset.smooth_spk(config.data.smth_std, name='smth')

    failed_trials = ~dataset.trial_info['is_successful'] 
    center_trials = dataset.trial_info['is_center_target']
    ol_block = session_csv.loc[session_csv['session_id'] == session, 'ol_blocks'].item() # cl = [int(i) for i in session_csv.loc[session_csv['session_id'] == session, 'cl_blocks'].item().split(' ')]
    cl_blocks =  ~dataset.trial_info['block_num'].isin([ol_block]).values.squeeze()
    
    trial_data = dataset.make_trial_data(
        align_field='start_time',
        align_range=(0, config.data.trial_len),
        allow_overlap=True,
        ignored_trials=failed_trials | center_trials | cl_blocks
    )

    trial_data.sort_index(axis=1, inplace=True)
    trial_data['X&Y'] = list(zip(trial_data['targetPos']['x'], trial_data['targetPos']['y']))
    trial_data['condition'] = 0

    # if cond_list == None:
    #     cond_list = list(zip(trial_data['X&Y'].unique(), np.arange(1,9)))
    cond_list = list(zip(trial_data['X&Y'].unique(), np.arange(1,9)))

    for xy, id in cond_list:    
        indices = trial_data.index[trial_data['X&Y'] == xy]
        trial_data.loc[indices, 'condition'] = id

    n_channels = trial_data.spikes.shape[-1]

    n_heldout = int(config.data.heldout_pct * n_channels)
    np.random.seed(config.setup.seed)
    heldout_channels = np.random.choice(n_channels, n_heldout, replace=False)
    heldin_channels = torch.ones(n_channels, dtype=bool)
    heldin_channels[heldout_channels] = False
    print(heldout_channels)

    avg_conds[session] = []
    trials_dict[session] = {}
    for cond_id, trials in trial_data.groupby('condition'):
        trial_list = []
        smth_trial_list = []
        for trial_id, trial in trials.groupby('trial_id'):
            # trial_list.append(trial.spikes.to_numpy())
            # smth_trial_list.append(trial.spikes_smth.to_numpy())
            trial_list.append(trial.spikes.to_numpy()[:, heldin_channels])
            smth_trial_list.append(trial.spikes_smth.to_numpy()[:, heldin_channels])
        trials_dict[session][cond_id] = smth_trial_list
        avg_conds[session].append(np.mean(smth_trial_list, 0))

session_list = list(avg_conds.keys())
avg_cond_arr = np.array(list(avg_conds.values())) # (days, conds, bins, chans)
avg_cond_arr = avg_cond_arr.transpose((3, 0, 1, 2)) # -> (chans, days, conds, bins)
nchans, ndays, nconds, nbins = avg_cond_arr.shape
avg_cond_arr = avg_cond_arr.reshape((nchans * ndays, nconds * nbins))

avg_cond_means = avg_cond_arr.mean(axis=1)
avg_cond_centered = (avg_cond_arr.T - avg_cond_means.T).T

pca = PCA(n_components=config.model.factor_dim)
pca.fit(avg_cond_centered.T)

dim_reduced_data = np.dot(avg_cond_centered.T, pca.components_.T).T
avg_cond_arr = avg_cond_arr.reshape((nchans, ndays, nconds, nbins))

dim_reduced_data_means = dim_reduced_data.mean(axis=1)
dim_reduced_data_this = (dim_reduced_data.T - dim_reduced_data_means.T).T

alignment_matrices = []
alignment_biases = []
for day in range(ndays):
    this_dataset_data = avg_cond_arr.reshape((nchans, ndays, nconds * nbins))[:, day, :].squeeze()

    this_dataset_means = avg_cond_means.reshape(nchans, ndays)[:, day].squeeze()
    this_dataset_centered = (this_dataset_data.T - this_dataset_means.T).T

    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(this_dataset_centered.T, dim_reduced_data_this.T)

    alignment_matrices.append(np.copy(reg.coef_.astype(np.float32)))  # nchans x nPCs
    bias = -1 * np.dot(this_dataset_means, reg.coef_.T)
    alignment_biases.append(bias.astype(np.float32))


import wandb

# with wandb.init(project='Alignment-Verification', name='All Channel PCs') as run:
with wandb.init(project='Alignment Bug Check', name='Before') as run:
# with wandb.init(project='Alignment-Verification', name='Heldin Channel PCs') as run:

    # # PLOT COND AVG


    fig = go.Figure()
    for idx, session in enumerate(session_list):
        avg_cond_arr = np.array(list(avg_conds[session])) #(conds, bins, chans)
        for condi, cond in enumerate(avg_cond_arr):
            cond = np.dot(cond, alignment_matrices[idx].T)
            cond = cond + alignment_biases[idx]

            fig.add_trace(
                go.Scatter3d(
                    x=cond[:, 0], 
                    y=cond[:, 1], 
                    z=cond[:, 2],
                    mode='lines',
                    line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
                )
            )

    fig.update_layout(
        width=430,
        height=410,
        autosize=False,
        showlegend=False,
        title={
            'text': "Condition Averaged PCs",
            'y':0.96,
            'yanchor': 'bottom'
        },
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(
                center=dict(
                    x=0.065,
                    y=0.0,
                    z=-0.075,
                    # z=-0.12,
                ),
                eye=dict(
                    x=1.3, 
                    y=1.3, 
                    z=1.3
                )
            ),
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual'
        ),
    )

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=20))
    config = {'displayModeBar': False}
    html_string = fig.to_html(config=config)


    wandb.log({'PC_plots_cond_avg': wandb.Html(html_string, inject=False)})

    # fig = go.Figure()
    # for idx, session in enumerate(session_list):
    #     avg_cond_arr = np.array(list(avg_conds[session])) #(conds, bins, chans)
    #     for condi, cond in enumerate(avg_cond_arr):
    #         cond = np.dot(cond, alignment_matrices[idx].T)
    #         cond = cond + alignment_biases[idx]

    #         fig.add_trace(
    #             go.Scatter3d(
    #                 x=cond[:, 3], 
    #                 y=cond[:, 4], 
    #                 z=cond[:, 5],
    #                 mode='lines',
    #                 line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
    #             )
    #         )

    # fig.update_layout(
    #     width=430,
    #     height=410,
    #     autosize=False,
    #     showlegend=False,
    #     title={
    #         'text': "Condition Averaged PCs",
    #         'y':0.96,
    #         'yanchor': 'bottom'
    #     },
    #     scene=dict(
    #         xaxis_showspikes=False,
    #         yaxis_showspikes=False,
    #         zaxis_showspikes=False,
    #         xaxis_title="PC3",
    #         yaxis_title="PC4",
    #         zaxis_title="PC5",
    #         camera=dict(
    #             center=dict(
    #                 x=0.065,
    #                 y=0.0,
    #                 z=-0.075,
    #                 # z=-0.12,
    #             ),
    #             eye=dict(
    #                 x=1.3, 
    #                 y=1.3, 
    #                 z=1.3
    #             )
    #         ),
    #         aspectratio = dict( x=1, y=1, z=1 ),
    #         aspectmode = 'manual'
    #     ),
    # )

    # fig.update_layout(margin=dict(r=0, l=0, b=0, t=20))
    # config = {'displayModeBar': False}
    # html_string = fig.to_html(config=config)


    # wandb.log({'PC_plots_cond_avg 2nd PCs': wandb.Html(html_string, inject=False)})

    # fig.write_html(f"Cond_avg_PCs.html", config=config)



    # # PLOT SINGLE TRIAL COND AVG


    # session_list_new = []
    # for idx, session in enumerate(session_list):
    #     cond_list = []
    #     for condi in trials_dict[session]:
    #         tr_list = []
    #         for trial in trials_dict[session][condi]:
    #             trial = np.dot(trial, alignment_matrices[idx].T)
    #             trial = trial + alignment_biases[idx]
    #             tr_list.append(trial)
    #         cond_list.append(np.mean(tr_list, 0))
    #     session_list_new.append(cond_list)


    # fig = go.Figure()
    # for session in session_list_new:
    #     for condi, cond in enumerate(session):
    #         print(cond.shape)
    #         fig.add_trace(
    #             go.Scatter3d(
    #                 x=cond[:, 0], 
    #                 y=cond[:, 1], 
    #                 z=cond[:, 2],
    #                 mode='lines',
    #                 line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
    #             )
    #         )

    # fig.update_layout(
        # width=430,
        # height=410,
    #     autosize=False,
    #     showlegend=False,
    #     title={
    #         'text': "Multisession Single Trial PCs",
    #         'y':0.96,
    #         'yanchor': 'bottom'
    #     },
    #     scene=dict(
    #         xaxis_showspikes=False,
    #         yaxis_showspikes=False,
    #         zaxis_showspikes=False,
    #         xaxis_title="PC1",
    #         yaxis_title="PC2",
    #         zaxis_title="PC3",
    #         camera=dict(
    #             center=dict(
    #                 x=0.065,
    #                 y=0.0,
    #                 z=-0.075,
    #                 # z=-0.12,
    #             ),
    #             eye=dict(
    #                 x=1.3, 
    #                 y=1.3, 
    #                 z=1.3
    #             )
    #         ),
    #         aspectratio = dict( x=1, y=1, z=1 ),
    #         aspectmode = 'manual'
    #     ),
    # )

    # fig.update_layout(margin=dict(r=0, l=0, b=0, t=20))
    # config = {'displayModeBar': False}

    # html_string = fig.to_html(config=config)

    # wandb.log({f'PC_plots_single_trial_cond_avg': wandb.Html(html_string, inject=False)})

    # fig.write_html(f"Session_avg_single_trial_PCs.html", config=config)



    # PLOT SINGLE TRIAL

    fig = go.Figure()
    for idx, session in enumerate(session_list):
        for condi in trials_dict[session]:
            for trial in trials_dict[session][condi]:
                trial = np.dot(trial, alignment_matrices[idx].T)
                trial = trial + alignment_biases[idx]

                fig.add_trace(
                    go.Scatter3d(
                        x=trial[:, 0], 
                        y=trial[:, 1], 
                        z=trial[:, 2],
                        mode='lines',
                        line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
                    )
                )

    fig.update_layout(
        width=430,
        height=410,
        autosize=False,
        showlegend=False,
        title={
            'text': "Single Trial PCs",
            'y':0.96,
            'yanchor': 'bottom'
        },
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(
                center=dict(
                    x=0.065,
                    y=0.0,
                    z=-0.075,
                    # z=-0.12,
                ),
                eye=dict(
                    x=1.3, 
                    y=1.3, 
                    z=1.3
                )
            ),
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual'
        ),
    )

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=20))
    config = {'displayModeBar': False}

    html_string = fig.to_html(config=config)
    # with wandb.init(project='example-sweep', name=f'PC_{session}') as run:
    #     wandb.log({f'PC_{session}': wandb.Html(html_string, inject=False)})
    wandb.log({f'PC_plots_single_trial': wandb.Html(html_string, inject=False)})

    #     fig.write_html(f"Session_{session}_single_trial_PCs.html", config=config)



    # PLOT PSTH
        
    # channel = 0

    # session_list_new = []
    # for idx, session in enumerate(session_list):
    #     cond_list = []
    #     for condi in trials_dict[session]:
    #         if condi <= 3:
    #             tr_list = []
    #             for trial in trials_dict[session][condi]:
    #                 tr_list.append(trial[:, channel])
    #             cond_list.append(np.mean(tr_list, 0))
    #     session_list_new.append(cond_list)

    # nrows = math.ceil(len(session_list) / 3)
    # ncols = 3

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 4))

    # ids = 0
    # for irow in range(nrows):
    #     for icol in range(ncols):
    #         ax = axes[irow, icol]
    #         if ids < len(session_list):
    #             ax.set_title(session_list[ids])
    #             for condi, cond in enumerate(session_list_new[ids]):
    #                 ax.plot(cond, color=cm.tab10(condi))
    #             ids += 1

    # st = fig.suptitle(f'Condition 1-3 PSTHs for Channel {channel}')
    # plt.savefig(f'cond_1-3_chan_{channel}_PSTH.png', bbox_extra_artists=[st], bbox_inches='tight')

    # wandb.log({f'Channel_{channel}_cond_1-3_PSTHs': wandb.Image(f'cond_1-3_chan_{channel}_PSTH.png')})








