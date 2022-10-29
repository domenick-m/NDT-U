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
from utils_f import get_config_from_file, set_seeds
from datasets import get_trial_data, chop, smooth_spikes
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import shutil
import os
import sys
import wandb


def test(config, model):

    model.eval()

    device = torch.device('cuda:0')
    set_seeds(config)

    # loop through data and run model on trials, need to get this from datasets
    trials_dict = get_trial_data()
    with torch.no_grad():
        outputs = {} # sessions
        for session in config.data.pretrain_sessions:
            outputs[session] = [] # conditions
            for condi, trials in trials_dict[session].items():
                outputs[session].append([]) #trials
                for trial in trials:
                    outputs[session][condi - 1].append(trial)

        # print(np.array(outputs).shape) (2, 8, 4, 200, 192) (sess, cond, tr, time, ch)  
        
        # plot each session pre-readout outputs colored by condition
        factors = []
        for name, session in outputs.items():
            for condi, condition in enumerate(session):
                for trial in condition:
                    inp = chop(np.array(trial), 30, 29)
                    inp = torch.Tensor(inp).to(device)
                    _, output = model(inp, [name for i in range(inp.shape[0])])
                    pre_smoothed_output = output[:, -1, :].cpu().numpy()
                    output = smooth_spikes(pre_smoothed_output, 60, 10, True)
                    factors.append(output)
        factors = np.array(factors)
        fs = factors.shape
        factors = factors.reshape((fs[0] * fs[1], fs[2]))
        pca = PCA(n_components=3)
        pca.fit(factors)

        for name, session in outputs.items():
            fig = go.Figure()
            for condi, condition in enumerate(session):
                for trial in condition:
                    inp = chop(np.array(trial), 30, 29)
                    inp = torch.Tensor(inp).to(device)
                    _, output = model(inp, [name for i in range(inp.shape[0])])
                    pre_smoothed_output = output[:, -1, :].cpu().numpy()
                    output = smooth_spikes(pre_smoothed_output, 60, 10, False)
                    low_d_factors = pca.transform(output)
                    fig.add_trace(
                        go.Scatter3d(
                            x=low_d_factors[:, 0], 
                            y=low_d_factors[:, 1], 
                            z=low_d_factors[:, 2],
                            mode='lines',
                            line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
                        )
                    )
            fig.update_layout(
                width=460,
                height=500,
                autosize=False,
                showlegend=False,
                title={
                    'text': "Multisession Condition Avg PCs",
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
            # html_string = fig.to_html(config=config)


            # wandb.log({'PC_plots_cond_avg': wandb.Html(html_string, inject=False)})

            fig.write_html(f"{name}_PCs.html", config=config)
        


def test():
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

    with wandb.init(project='Alignment-Verification', name='PCR readin 64dim no xcorr') as run:

        # loop through data and run model on trials, need to get this from datasets
        trials_dict = get_testing_data(config)
        with torch.no_grad():
            outputs = {} # sessions
            for session in config.data.pretrain_sessions:
                outputs[session] = [] # conditions
                for condi, trials in trials_dict[session].items():
                    outputs[session].append([]) #trials
                    for trial in trials:
                        outputs[session][condi - 1].append(trial)

            # print(np.array(outputs).shape) (2, 8, 4, 200, 192) (sess, cond, tr, time, ch)  
            
            # plot each session pre-readout outputs colored by condition
            factors = []
            for name, session in outputs.items():
                for condi, condition in enumerate(session):
                    for trial in condition:
                        inp = chop(np.array(trial), 30, 29)
                        inp = torch.Tensor(inp).to(device)
                        # output, _ = model(inp, [name for i in range(inp.shape[0])])
                        spikes = inp[:, -1, 5:]
                        rates, output = model(inp, [name for i in range(inp.shape[0])])
                        s_saved = inp[0, :, 5:]
                        r_saved = rates[0, :, 5:]
                        rates = rates[:, -1, 5:]
                        pre_smoothed_output = output[:, -1, :].cpu().numpy()
                        output = smooth_spikes(pre_smoothed_output, config['data']['smth_std'], 10, False)
                        factors.append(output)

            plt.plot(smooth_spikes(spikes[:, 0].cpu().numpy(), 60, 10, False), label="Smooth Spikes")
            plt.plot(rates[:, 0].exp().cpu().numpy(), label="NDT Rates")
            plt.legend()
            plt.xlabel("Time (10ms bins)")
            plt.ylabel("Spks / sec")
            plt.tight_layout()
            plt.savefig(f"images/{name}_trial_rates_v_spikes_60ms.png")
            plt.close()

            # plt.plot(smooth_spikes(s_saved[29:, 0].cpu().numpy(), 60, 10, False))
            # plt.plot(r_saved[:, 0].exp().cpu().numpy())
            # plt.tight_layout()
            # plt.savefig(f"images/{name}_rates_v_spikes_60ms.png")
            # plt.close()

            wandb.log({
                "trial_rates_v_smth_spikes_60ms": wandb.Image(f"images/{name}_trial_rates_v_spikes_60ms.png"),
                # "rates_v_smth_spikes_60ms": wandb.Image(f"images/{name}_rates_v_spikes_60ms.png"),
            })

            factors = np.array(factors)
            fs = factors.shape
            factors = factors.reshape((fs[0] * fs[1], fs[2]))
            pca = PCA(n_components=3)
            pca.fit(factors)

            # for name, session in outputs.items():
            #     fig = go.Figure()
            #     for condi, condition in enumerate(session):
            #         for trial in condition:
            #             inp = chop(np.array(trial), 30, 29)
            #             inp = torch.Tensor(inp).to(device)
            #             # output, _ = model(inp, [name for i in range(inp.shape[0])])
            #             _, output = model(inp, [name for i in range(inp.shape[0])])
            #             pre_smoothed_output = output[:, -1, :].cpu().numpy()
            #             output = smooth_spikes(pre_smoothed_output, config['data']['smth_std'], 10, False)
            #             low_d_factors = pca.transform(output)
            #             fig.add_trace(
            #                 go.Scatter3d(
            #                     x=low_d_factors[:, 0], 
            #                     y=low_d_factors[:, 1], 
            #                     z=low_d_factors[:, 2],
            #                     mode='lines',
            #                     line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
            #                 )
            #             )
            #     fig.update_layout(
            #         width=460,
            #         height=500,
            #         autosize=False,
            #         showlegend=False,
            #         title={
            #             'text': "Multisession Condition Avg PCs",
            #             'y':0.96,
            #             'yanchor': 'bottom'
            #         },
            #         scene=dict(
            #             xaxis_showspikes=False,
            #             yaxis_showspikes=False,
            #             zaxis_showspikes=False,
            #             xaxis_title="PC1",
            #             yaxis_title="PC2",
            #             zaxis_title="PC3",
            #             camera=dict(
            #                 center=dict(
            #                     x=0.065,
            #                     y=0.0,
            #                     z=-0.075,
            #                     # z=-0.12,
            #                 ),
            #                 eye=dict(
            #                     x=1.3, 
            #                     y=1.3, 
            #                     z=1.3
            #                 )
            #             ),
            #             aspectratio = dict( x=1, y=1, z=1 ),
            #             aspectmode = 'manual'
            #         ),
            #     )

            #     fig.update_layout(margin=dict(r=0, l=0, b=0, t=20))
            #     config2 = {'displayModeBar': False}
            #     html_string = fig.to_html(config=config)


            #     wandb.log({f"{name}_PCs": wandb.Html(html_string, inject=False)})

            #     fig.write_html(f"{name}_PCs.html", config=config2)

            fig = go.Figure()
            for name, session in outputs.items():
                for condi, condition in enumerate(session):
                    for trial in condition:
                        inp = chop(np.array(trial), 30, 29)
                        inp = torch.Tensor(inp).to(device)
                        _, output = model(inp, [name for i in range(inp.shape[0])])
                        pre_smoothed_output = output[:, -1, :].cpu().numpy()
                        output = smooth_spikes(pre_smoothed_output, config['data']['smth_std'], 10, False)
                        low_d_factors = pca.transform(output)
                        fig.add_trace(
                            go.Scatter3d(
                                x=low_d_factors[:, 0], 
                                y=low_d_factors[:, 1], 
                                z=low_d_factors[:, 2],
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
            config2 = {'displayModeBar': False}
            html_string = fig.to_html(config=config2)
            wandb.log({f"single_trial_PCs.html": wandb.Html(html_string, inject=False)})


            cond_avs = []
            for name, session in outputs.items():
                sess_av = []
                for condi, condition in enumerate(session):
                    cond_av = []
                    for trial in condition:
                        inp = chop(np.array(trial), 30, 29)
                        inp = torch.Tensor(inp).to(device)
                        _, output = model(inp, [name for i in range(inp.shape[0])])
                        pre_smoothed_output = output[:, -1, :].cpu().numpy()
                        output = smooth_spikes(pre_smoothed_output, config['data']['smth_std'], 10, False)
                        low_d_factors = pca.transform(output)
                        cond_av.append(low_d_factors)
                    sess_av.append(np.array(cond_av).mean(0))
                cond_avs.append(sess_av)

            fig = go.Figure()
            for session in cond_avs:
                for condi, condition in enumerate(session):
                    fig.add_trace(
                        go.Scatter3d(
                            x=condition[:, 0], 
                            y=condition[:, 1], 
                            z=condition[:, 2],
                            mode='lines',
                            line=dict(color=f'{colors.rgb2hex(cm.tab10(condi))}'),
                        )
                    )
            fig.update_layout(
                width=460,
                height=500,
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
        
if __name__ == "__main__":
    test()
