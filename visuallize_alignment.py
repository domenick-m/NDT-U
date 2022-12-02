# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('../')
from utils.multisession_utils import align_sessions
from utils.config_utils import get_config
from utils.plot.pcs import plot_pcs
import numpy as np
from utils.t5_utils import load_toolkit_datasets, get_trialized_data
config = get_config()
## Start Time
config.defrost()
config.data.ol_align_field = 'start_time'
config.data.ol_align_range = [0, 2500]
config.freeze()
alignment_matrices, alignment_biases = align_sessions(config)
datasets = load_toolkit_datasets(config)
config.defrost()
config.data.ol_align_field = 'start_time'
config.data.ol_align_range = [-0, 2500]
config.data.cl_align_field = 'start_time'
config.data.cl_align_range = [0, 2000]
config.freeze()
trialized_data = get_trialized_data(config, datasets)
ol_cond_avg = ([], []) 
ol_single_trial = ([], [])
cl_single_trial = ([], [])

trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size

for idx, session in enumerate(config.data.sessions):    
    for cond_id, trials in trialized_data[session]['ol_trial_data'].groupby('condition'):
        if cond_id != 0:
            low_d_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                heldin_spikes = trial.spikes_smth.to_numpy()[:, datasets[session].heldin_channels]
                if heldin_spikes.shape[0] == trial_len:
                    low_d_trial = np.dot(heldin_spikes, alignment_matrices[idx].T)
                    low_d_trial = low_d_trial + np.array(alignment_biases[idx])
                    low_d_trials.append(low_d_trial)

            ol_single_trial[0].append(np.concatenate(low_d_trials, 0))
            ol_single_trial[1].append(cond_id)

            ol_cond_avg[0].append(np.array(low_d_trials).mean(0))
            ol_cond_avg[1].append(cond_id)

    for cond_id, trials in trialized_data[session]['cl_trial_data'].groupby('condition'):
        if cond_id != 0:
            low_d_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                if trial.shape[0] > 45 and trial.shape[0] < 400:
                    heldin_spikes = trial.spikes_smth.to_numpy()[45:1795, datasets[session].heldin_channels]
                    
                    low_d_trial = np.dot(heldin_spikes, alignment_matrices[idx].T)
                    low_d_trial = low_d_trial + np.array(alignment_biases[idx])
                    low_d_trials.append(low_d_trial)

            cl_single_trial[0].append(np.concatenate(low_d_trials, 0))
            cl_single_trial[1].append(cond_id)

print('MAKING!')
from matplotlib import cm, colors

from matplotlib import cm, colors

import time
import numpy as np

# Define frames
import plotly.graph_objects as go

norm = colors.Normalize(vmin=0, vmax=8, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
mapper.set_array([])

pcs = ol_cond_avg[0]

pcs = [trial.astype(np.float16) for trial in pcs]
conds = len(pcs)
# nb_frames = pcs[0].shape[0]
nb_frames = 30
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pcs[i][:1, 0], 
            y=pcs[i][:1, 1], 
            z=pcs[i][:1, 2],
            mode='lines',
            line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][i]))}'),
        ) for i in range(conds)
    ],
    frames=[
        go.Frame(
            data=[
                go.Scatter3d(
                    x=pcs[i][:k*5, 0], 
                    y=pcs[i][:k*5, 1], 
                    z=pcs[i][:k*5, 2],
                    mode='lines',
                    line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][i]))}'),
                ) for i in range(conds)
            ],
            name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)
    ]
)

# Add data to be displayed before animation starts
fig.add_trace(go.Scatter3d(
        x=pcs[0][:, 0], 
        y=pcs[0][:, 1], 
        z=pcs[0][:, 2],
        mode='lines',
        line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][0]))}'),
    ))


def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='Slices in volumetric data',
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[min(pcs[0][:,2])-0.3, max(pcs[0][:,2])+0.3], autorange=False),
                    xaxis=dict(range=[min(pcs[0][:,0])-0.3, max(pcs[0][:,0])+0.3], autorange=False),
                    yaxis=dict(range=[min(pcs[0][:,1])-0.3, max(pcs[0][:,1])+0.3], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(200)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)

fig.write_html('test.html', full_html=False, include_plotlyjs='cdn')
exit()
fig = plot_pcs(*ol_cond_avg, 'OL Condition Averaged', return_fig=True)
fig.show()
# print ya
# fig = plot_pca(*ol_single_trial, 'OL Single Trial', return_fig=True)
# fig.show()
# fig = plot_pca(*cl_single_trial, 'CL Single Trial', return_fig=True)
# fig.show()
## Speed Onset
config.defrost()
config.data.ol_align_field = 'speed_onset'
config.data.ol_align_range = [-700, 1250]
config.freeze()
so_alignment_matrices, so_alignment_biases = align_sessions(config)
config.defrost()
config.data.ol_align_field = 'speed_onset'
config.data.ol_align_range = [-700, 1250]
config.data.cl_align_field = 'start_time'
config.data.cl_align_range = [500, 2000]
config.freeze()
trialized_data = get_trialized_data(config, datasets)
ol_cond_avg = ([], []) 
ol_single_trial = ([], [])
cl_single_trial = ([], [])

trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size
cl_trial_len = (config.data.cl_align_range[1] - config.data.cl_align_range[0]) / config.data.bin_size

for idx, session in enumerate(config.data.sessions):    
    for cond_id, trials in trialized_data[session]['ol_trial_data'].groupby('condition'):
        if cond_id != 0:
        # if cond_id == 1:
            low_d_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                heldin_spikes = trial.spikes_smth.to_numpy()[:, datasets[session].heldin_channels]
                if heldin_spikes.shape[0] == trial_len:
                    low_d_trial = np.dot(heldin_spikes[75:], so_alignment_matrices[idx].T)
                    low_d_trial = low_d_trial + np.array(so_alignment_biases[idx])
                    low_d_trials.append(low_d_trial)

            ol_single_trial[0].append(np.concatenate(low_d_trials, 0))
            ol_single_trial[1].append(cond_id)

            ol_cond_avg[0].append(np.array(low_d_trials).mean(0))
            ol_cond_avg[1].append(cond_id)

    for cond_id, trials in trialized_data[session]['cl_trial_data'].groupby('condition'):
        if cond_id != 0:
            low_d_trials = []
            for trial_id, trial in trials.groupby('trial_id'):
                # if trial.shape[0] > 45 and trial.shape[0] < 400:
                heldin_spikes = trial.spikes_smth.to_numpy()[:, datasets[session].heldin_channels]
                # if heldin_spikes.shape[0] == trial_len:
                
                low_d_trial = np.dot(heldin_spikes, so_alignment_matrices[idx].T)
                low_d_trial = low_d_trial + np.array(so_alignment_biases[idx])
                low_d_trials.append(low_d_trial)

            cl_single_trial[0].append(np.concatenate(low_d_trials, 0))
            cl_single_trial[1].append(cond_id)
# from matplotlib import cm, colors

# import time
# import numpy as np

# # Define frames
# import plotly.graph_objects as go

# norm = colors.Normalize(vmin=0, vmax=8, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
# mapper.set_array([])

# pcs = ol_cond_avg[0]
# conds = len(pcs)
# # nb_frames = pcs[0].shape[0]
# nb_frames = 60
# fig = go.Figure(
#     data=[
#         go.Scatter3d(
#             x=pcs[i][:1, 0], 
#             y=pcs[i][:1, 1], 
#             z=pcs[i][:1, 2],
#             mode='lines',
#             line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][i]))}'),
#         ) for i in range(conds)
#     ],
#     frames=[
#         go.Frame(
#             data=[
#                 go.Scatter3d(
#                     x=pcs[i][:k*5, 0], 
#                     y=pcs[i][:k*5, 1], 
#                     z=pcs[i][:k*5, 2],
#                     mode='lines',
#                     line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][i]))}'),
#                 ) for i in range(conds)
#             ],
#             name=str(k) # you need to name the frame for the animation to behave properly
#         )
#         for k in range(nb_frames)
#     ]
# )

# # Add data to be displayed before animation starts
# fig.add_trace(go.Scatter3d(
#         x=pcs[0][:, 0], 
#         y=pcs[0][:, 1], 
#         z=pcs[0][:, 2],
#         mode='lines',
#         line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(ol_cond_avg[1][0]))}'),
#     ))


# def frame_args(duration):
#     return {
#             "frame": {"duration": duration},
#             "mode": "immediate",
#             "fromcurrent": True,
#             "transition": {"duration": duration, "easing": "linear"},
#         }

# sliders = [
#             {
#                 "pad": {"b": 10, "t": 60},
#                 "len": 0.9,
#                 "x": 0.1,
#                 "y": 0,
#                 "steps": [
#                     {
#                         "args": [[f.name], frame_args(0)],
#                         "label": str(k),
#                         "method": "animate",
#                     }
#                     for k, f in enumerate(fig.frames)
#                 ],
#             }
#         ]

# # Layout
# fig.update_layout(
#          title='Slices in volumetric data',
#          width=600,
#          height=600,
#          scene=dict(
#                     zaxis=dict(range=[-2.75, 2.75], autorange=False),
#                     xaxis=dict(range=[-2.75, 2.75], autorange=False),
#                     yaxis=dict(range=[-2.75, 2.75], autorange=False),
#                     aspectratio=dict(x=1, y=1, z=1),
#                     ),
#          updatemenus = [
#             {
#                 "buttons": [
#                     {
#                         "args": [None, frame_args(200)],
#                         "label": "&#9654;", # play symbol
#                         "method": "animate",
#                     },
#                     {
#                         "args": [[None], frame_args(0)],
#                         "label": "&#9724;", # pause symbol
#                         "method": "animate",
#                     },
#                 ],
#                 "direction": "left",
#                 "pad": {"r": 10, "t": 70},
#                 "type": "buttons",
#                 "x": 0.1,
#                 "y": 0,
#             }
#          ],
#          sliders=sliders
# )

# fig.show()
fig = plot_pcs(*ol_cond_avg, 'OL Condition Averaged', return_fig=True)
fig.show()
fig = plot_pca(*ol_single_trial, 'OL Single Trial', return_fig=True)
fig.show()
# fig = plot_pca(*cl_single_trial, 'CL Single Trial', return_fig=True)
# fig.show()