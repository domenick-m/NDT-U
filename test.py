from utils.data.create_local_t5data import get_trial_data
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
                    output = smooth_spikes(pre_smoothed_output, 60, 10, True)
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
        

