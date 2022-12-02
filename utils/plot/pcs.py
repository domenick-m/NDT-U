from matplotlib import cm, colors
import numpy as np
import plotly.graph_objects as go
import copy


def plot_pcs(pcs, conditions, title, return_fig=False, animate=False):
    # decrease storage footprint as much as possible
    pcs = [trial.astype(np.float16) for trial in pcs]
    
    norm = colors.Normalize(vmin=0, vmax=8, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
    mapper.set_array([])

    fig = go.Figure()

    for trial, cond_id in zip(pcs, conditions):
        fig.add_trace(
            go.Scatter3d(
                x=trial[:, 0], 
                y=trial[:, 1], 
                z=trial[:, 2],
                mode='lines',
                line=dict(color=f'{colors.rgb2hex(cm.tab10(cond_id-1))}'),
            )
        )

    fig.update_layout(
        width=460,
        height=500,
        autosize=False,
        showlegend=False,
        title={
            'text': title,
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

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=50))

    config = {'displayModeBar': False}
    return fig if return_fig else fig.to_html(config=config, full_html=False, include_plotlyjs='cdn') 
    
def plot_pcs_2(pcs, conditions, title, return_fig=False):
    # decrease storage footprint as much as possible
    pcs = [trial.astype(np.float16) for trial in pcs]
    
    norm = colors.Normalize(vmin=0, vmax=8, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
    mapper.set_array([])

    fig = go.Figure()

    for trial, cond_id in zip(pcs, conditions):
        fig.add_trace(
            go.Scatter3d(
                x=trial[:, 0], 
                y=trial[:, 1], 
                z=trial[:, 2],
                mode='lines',
                line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(cond_id))}'),
            )
        )

    fig.update_layout(
        width=460,
        height=500,
        autosize=False,
        showlegend=False,
        title={
            'text': title,
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

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=50))

    config = {'displayModeBar': False}
    return fig if return_fig else fig.to_html(config=config, full_html=False, include_plotlyjs='cdn') 
    

from matplotlib import cm, colors

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