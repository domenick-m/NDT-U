from create_local_t5data import get_trial_data
from datasets import get_testing_data
from utils_f import get_config
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from utils.plotting.plot_true_vs_pred_mvmnt import plot_true_vs_pred_mvmnt
import torch
from utils_f import get_config_from_file, set_seeds
from datasets import get_trial_data, chop, smooth_spikes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import shutil
import os
import sys
import wandb
import pandas as pd
import copy


def test_func(config, model):

    model.eval()
    name = model.name 

    device = torch.device('cuda:0')
    set_seeds(config)

    datasets = get_testing_data(config)
    session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

    trialized_data = {}
    for session in config.data.pretrain_sessions:
        dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

        session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

        if config.data.rem_xcorr: 
            corr, corr_chans = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)
            
        dataset.resample(config.data.bin_size / 1000)
        dataset.smooth_spk(20, name='smth') # for use if we want to take mean and std of smth values

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
            torch_rates, output = [], []
            for i in range(0, hi_chopped_spks.shape[0], 10):
                ret_tuple = model(hi_chopped_spks[i:i+10, :, :], names[i:i+10])
                torch_rates.append(ret_tuple[0])
                output.append(ret_tuple[1])
            rates = torch.cat(torch_rates, 0)
            output = torch.cat(output, 0)

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

        for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):
            indices = trial_data.index[trial_data['X&Y'] == xy]
            trial_data.loc[indices, 'condition'] = id
        
        trialized_data[session] = trial_data

    factors = []
    for session in config.data.pretrain_sessions:    
        for cond_id, trials in trialized_data[session].groupby('condition'):
            for trial_id, trial in trials.groupby('trial_id'):
                factors.append(trial.factors_smth.to_numpy())

    factors = np.array(factors)
    fs = factors.shape
    factors = factors.reshape((fs[0] * fs[1], fs[2]))
    pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
    pca.fit(factors)

    # COND AVG
    
    fig = go.Figure()
    for session in config.data.pretrain_sessions:    
        for cond_id, trials in trialized_data[session].groupby('condition'):
            avg_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                avg_trials.append(pca.transform(trial.factors_smth))
            avg_trials = np.array(avg_trials).mean(0)
            fig.add_trace(
                go.Scatter3d(
                    x=avg_trials[:, 0], 
                    y=avg_trials[:, 1], 
                    z=avg_trials[:, 2],
                    mode='lines',
                    line=dict(color=f'{colors.rgb2hex(cm.tab10(cond_id))}'),
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
    config2 = {'displayModeBar': False}
    html_string = fig.to_html(config=config2, full_html=False, include_plotlyjs='cdn')

    wandb.log({'PC_plots_cond_avg': wandb.Html(html_string, inject=False)})

    # SINGLE TRIAL

    fig = go.Figure()
    for session in config.data.pretrain_sessions:    
        for cond_id, trials in trialized_data[session].groupby('condition'):
            for trial_id, trial in trials.groupby('trial_id'):
                pc_factors = pca.transform(trial.factors_smth)
                fig.add_trace(
                    go.Scatter3d(
                        x=pc_factors[:, 0], 
                        y=pc_factors[:, 1], 
                        z=pc_factors[:, 2],
                        mode='lines',
                        line=dict(color=f'{colors.rgb2hex(cm.tab10(cond_id))}'),
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
    config2 = {'displayModeBar': False}
    html_string = fig.to_html(config=config2, full_html=False, include_plotlyjs='cdn')
    wandb.log({f"single_trial_PCs.html": wandb.Html(html_string, inject=False)})
        
        
def plot_xcorr_hist( xcorr_data, hist_binwidth=0.0001, title=None , i='error'):
    """
    plot histogram of xcorr counts
    """
    plt.clf()
    plt.Figure()
    plt.subplots(1, 1, figsize=(8, 6))
    data = np.vstack( xcorr_data )
    # data = xcorr_data 
    plt.hist( data, bins=np.arange( min(data), max(data) + hist_binwidth, hist_binwidth ), edgecolor='black')
    plt.yscale( 'log' )
    plt.xlabel( 'Cross-Correlation' )
    plt.ylabel( 'Counts' )
    if title is not None:
        plt.title( title )
    plt.axvline(0.2, color='k', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'xcorr_test_{i}.png', facecolor='white', transparent=False)
    plt.close()  

def test(path=None):
    if path == None:
        if len(sys.argv) == 1 or len(sys.argv) > 2:
            print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
            exit()
        path = sys.argv[1]
    name = path[:path.rindex('/')].split('/')[-1]
    config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')
    if not os.path.isdir(f"plots/{name}"): os.makedirs(f"plots/{name}")
    shutil.copyfile(path[:path.rindex('/')+1]+'config.yaml', f"plots/{name}/config.yaml")
    device = torch.device('cuda:0')

    set_seeds(config)

    model = torch.load(path).to(device)
    model.name = name

    model.eval()

    device = torch.device('cuda:0')
    set_seeds(config)

    with wandb.init(project='Alignment PSTH Visualization Check', name='honest-sweep-10') as run:

        # trials_dict, datasets = get_testing_data(config)
        datasets = get_testing_data(config)
        session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

        session_rates_mean_rem_div = {}
        session_rates_mean_rem = {}
        session_rates = {}
        session_means = {}
        session_factors = {}

        trialized_data = {}
        for session in config.data.pretrain_sessions:
            dataset = copy.deepcopy(datasets[session]) # do not want to run xcorr on test data

            session_csv = pd.read_csv(f'{config.data.dir}/sessions.csv')

            if config.data.rem_xcorr: 
                corr, corr_chans, pairs = dataset.get_pair_xcorr('spikes', threshold=0.2, zero_chans=True)
                # plot_xcorr_hist(pairs, i=f'{session}', hist_binwidth=0.005, title=f'{session}')
               
            dataset.resample(config.data.bin_size / 1000)
            dataset.smooth_spk(20, name='smth') # for use if we want to take mean and std of smth values

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
            
            mean = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.mean(0).to_numpy()
            std = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes_smth.std(0).to_numpy()
            # mean = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.mean(0).to_numpy()
            # std = dataset.data[dataset.data['blockNums'].isin([ol_block]).values.squeeze()].spikes.std(0).to_numpy()

            chopped_spks = chop(np.array(spks[:, heldin_channels]), 30, 29)
            hi_chopped_spks = torch.Tensor(chopped_spks).to(device)

            names = [session for i in range(hi_chopped_spks.shape[0])]
            with torch.no_grad():
                rates, output = model(hi_chopped_spks, names)

            chans = []
            for ch in corr_chans:
                if 'ch00' in ch:
                    chans.append(int(ch.split('ch00')[1]))
                elif 'ch0' in ch:
                    chans.append(int(ch.split('ch0')[1]))
                elif 'ch' in ch:
                    chans.append(int(ch.split('ch')[1]))
            not_corr_channels = torch.ones(rates.shape[-1], dtype=bool)
            not_corr_channels[chans] = False

            heldin_not_corr_idxs = not_corr_channels[heldin_channels]
            heldout_not_corr_idxs = not_corr_channels[heldout_channels]
            all_not_corr_channels = np.expand_dims(np.expand_dims(np.concatenate((heldin_not_corr_idxs, heldout_not_corr_idxs), -1), 0), 0)

            rates = rates.exp().cpu().numpy()
            heldin_rates_saved = rates[:, :, :n_heldin]
            heldout_rates_saved = rates[:, :, n_heldin:]

            heldin_rates_mean = np.expand_dims(mean[heldin_channels], 0)
            heldout_rates_mean = np.expand_dims(mean[heldout_channels], 0)
            heldin_rates_std = np.expand_dims(std[heldin_channels], 0)
            heldout_rates_std = np.expand_dims(std[heldout_channels], 0)

            heldin_rates_mean_rem = heldin_rates_saved - heldin_rates_mean
            heldout_rates_mean_rem = heldout_rates_saved - heldout_rates_mean

            heldin_rates_mean_rem_div = heldin_rates_mean_rem / heldin_rates_std
            heldout_rates_mean_rem_div = heldout_rates_mean_rem / heldout_rates_std

            all_rates_mean_rem_div = np.concatenate((heldin_rates_mean_rem_div, heldout_rates_mean_rem_div), -1)
            ar_shape = all_rates_mean_rem_div.shape
            all_rates_mean_rem_div_no_corr = all_rates_mean_rem_div[np.repeat(np.repeat(all_not_corr_channels, ar_shape[1], 1), ar_shape[0], 0)]

            session_factors[session] = output.cpu().numpy()
            session_rates_mean_rem_div[session] = all_rates_mean_rem_div_no_corr.reshape((ar_shape[0], ar_shape[1], -1))
            
            hi_shape = heldin_rates_mean_rem.shape
            
            heldin_not_corr_idxs_exp = np.repeat(np.repeat(heldin_not_corr_idxs, hi_shape[1], 1), hi_shape[0], 0)
            heldout_not_corr_idxs_exp = np.repeat(np.repeat(heldin_not_corr_idxs, hi_shape[1], 1), hi_shape[0], 0)
            
            

            session_rates_mean_rem[session] = np.expand_dims(np.concatenate(
                (heldin_rates_mean_rem[heldin_not_corr_idxs_exp], heldout_rates_mean_rem[heldout_not_corr_idxs_exp]), 
                -1
            ), 0)


            r_shape = rates.shape
            session_rates[session] = rates[np.repeat(np.repeat(all_not_corr_channels, r_shape[1], 1), r_shape[0], 0)].reshape((r_shape[0], r_shape[1], -1))
            session_means[session] = np.expand_dims(np.concatenate(
                (heldin_rates_mean[np.expand_dims(heldin_not_corr_idxs, 0)], heldout_rates_mean[np.expand_dims(heldout_not_corr_idxs, 0)]), 
                -1
            ), 0)

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

            for xy, id in list(zip(trial_data['X&Y'].unique(), np.arange(1,9))):
                indices = trial_data.index[trial_data['X&Y'] == xy]
                trial_data.loc[indices, 'condition'] = id
            
            trialized_data[session] = trial_data

        # for i in range(session_rates[config.data.pretrain_sessions[0]].shape[0]):
        #     plt.pcolormesh(session_rates[config.data.pretrain_sessions[0]][i].T, cmap = 'Reds' )
        #     plt.tight_layout()
        #     plt.savefig(f"images/test/{model.name}_rates_{i}.png")
        #     plt.close()


        plt.pcolormesh(session_factors[config.data.pretrain_sessions[0]][0].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_factors.png")
        plt.close()

        plt.pcolormesh(session_rates_mean_rem[config.data.pretrain_sessions[0]][0].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_means_rem.png")
        plt.close()

        plt.pcolormesh(session_factors[config.data.pretrain_sessions[0]][0].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_factors.png")
        plt.close()

        plt.pcolormesh(session_factors[config.data.pretrain_sessions[0]][37].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_factors_37.png")
        plt.close()

        plt.pcolormesh(session_rates_mean_rem_div[config.data.pretrain_sessions[0]][0].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_rates_mean_rem_div.png")
        plt.close()

        plt.pcolormesh(session_rates_mean_rem_div[config.data.pretrain_sessions[0]][37].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_rates_mean_rem_div_37.png")
        plt.close()

        plt.pcolormesh(session_rates[config.data.pretrain_sessions[0]][0].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_rates.png")
        plt.close()

        plt.pcolormesh(session_rates[config.data.pretrain_sessions[0]][37].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_rates_37.png")
        plt.close()

        plt.pcolormesh(session_means[config.data.pretrain_sessions[0]].T, cmap = 'Reds' )
        plt.tight_layout()
        plt.savefig(f"images/{model.name}_means.png")
        plt.close()

        wandb.log({
            "rates": wandb.Image(f"images/{model.name}_rates.png"),
            "factors": wandb.Image(f"images/{model.name}_factors.png"),
            "means": wandb.Image(f"images/{model.name}_means.png"),
            "rates_mean_rem_div": wandb.Image(f"images/{model.name}_rates_mean_rem_div.png"),
            "rates_37": wandb.Image(f"images/{model.name}_rates_37.png"),
            "factors_37": wandb.Image(f"images/{model.name}_factors_37.png"),
            "rates_mean_rem_div_37": wandb.Image(f"images/{model.name}_rates_mean_rem_div_37.png"),
        })

        factors = []
        for session in config.data.pretrain_sessions:    
            for cond_id, trials in trialized_data[session].groupby('condition'):
                for trial_id, trial in trials.groupby('trial_id'):
                    factors.append(trial.factors_smth)

        factors = np.array(factors)
        fs = factors.shape
        factors = factors.reshape((fs[0] * fs[1], fs[2]))
        pca = PCA(n_components=3)
        # pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
        pca.fit(factors)

        # COND AVG
            
        fig = go.Figure()
        for session in config.data.pretrain_sessions:    
            for cond_id, trials in trialized_data[session].groupby('condition'):
                avg_trials = []
                for trial_id, trial in trials.groupby('trial_id'):
                    avg_trials.append(pca.transform(trial.factors_smth))
                avg_trials = np.array(avg_trials).mean(0)
                fig.add_trace(
                    go.Scatter3d(
                        x=avg_trials[:, 0], 
                        y=avg_trials[:, 1], 
                        z=avg_trials[:, 2],
                        mode='lines',
                        line=dict(color=f'{colors.rgb2hex(cm.tab10(cond_id))}'),
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
        config2 = {'displayModeBar': False}
        html_string = fig.to_html(config=config2)

        wandb.log({'PC_plots_cond_avg': wandb.Html(html_string, inject=False)})


        # SINGLE TRIAL

        fig = go.Figure()
        for session in config.data.pretrain_sessions:    
            for cond_id, trials in trialized_data[session].groupby('condition'):
                for trial_id, trial in trials.groupby('trial_id'):
                    pc_factors = pca.transform(trial.factors_smth)
                    fig.add_trace(
                        go.Scatter3d(
                            x=pc_factors[:, 0], 
                            y=pc_factors[:, 1], 
                            z=pc_factors[:, 2],
                            mode='lines',
                            line=dict(color=f'{colors.rgb2hex(cm.tab10(cond_id))}'),
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
        config2 = {'displayModeBar': False}
        html_string = fig.to_html(config=config2)

        wandb.log({'PC_plots_single_trial': wandb.Html(html_string, inject=False)})

        
if __name__ == "__main__":
    test()
