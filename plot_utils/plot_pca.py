from matplotlib import cm, colors
import numpy as np
import plotly.graph_objects as go
import copy


def plot_pca(
    pcs, 
    c_smth_pcs, 
    ac_smth_pcs, 
    angles, 
    smth_std, 
    add_legend,
    title
):

    pcs = [i.astype(np.float16) for i in pcs]
    c_smth_pcs = [i.astype(np.float16) for i in c_smth_pcs]
    ac_smth_pcs = [i.astype(np.float16) for i in ac_smth_pcs]
    
    norm = colors.Normalize(vmin=-180, vmax=180, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
    mapper.set_array([])

    fig = go.Figure()

    for trial, angle in zip(pcs, angles):
        fig.add_trace(
            go.Scatter3d(
                x=trial[:, 0], 
                y=trial[:, 1], 
                z=trial[:, 2],
                mode='lines',
                line=dict(color=f'{colors.rgb2hex(mapper.to_rgba(angle))}'),
            )
        )

    fig.update_layout(
        width=500,
        height=500,
        autosize=False,
        showlegend=False,
        title=title,
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(
                center=dict(
                    x=0.0,
                    y=0.0,
                    z=-0.125,
                ),
            ),
            aspectratio = dict( x=1, y=1, z=1 ),
            aspectmode = 'manual'
        ),
    )

    rates_smth_buttons = [
        {
            'method': 'update',
            'label': f'None',
            'visible': True,
            'args': [{
                'x': [i[:, 0] for i in pcs],
                'y': [i[:, 1] for i in pcs],
                'z': [i[:, 2] for i in pcs],
            }],
        },
        {
            'method': 'update',
            'label': f'Acausal ({smth_std}ms std)',
            'visible': True,
            'args': [{
                'x': [i[:, 0] for i in c_smth_pcs],
                'y': [i[:, 1] for i in c_smth_pcs],
                'z': [i[:, 2] for i in c_smth_pcs],
            }],
        },
        {
            'method': 'update',
            'label': f'Causal ({smth_std}ms std)',
            'visible': True,
            'args': [{
                'x': [i[:, 0] for i in ac_smth_pcs],
                'y': [i[:, 1] for i in ac_smth_pcs],
                'z': [i[:, 2] for i in ac_smth_pcs],
            }],
        },
    ]

    um = [
        {
            'buttons':rates_smth_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b": 0, "l": 0},
            'showactive':True,
            'x':0.6,
            'xanchor':"center",
            'y':1.01,
            'yanchor':"bottom"
        }
    ]

    annotations=[
            dict(
                text="Rates Smoothing:", 
                showarrow=False,
                x=0.1, 
                y=1.01, 
                yref="paper", 
                xref="paper",
                xanchor="left", 
                align="left",
            )
        ]

    fig.update_layout(updatemenus=um, annotations=annotations)

    if add_legend:
        fig.add_layout_image(
            dict(
                source="https://domenick-m.github.io/NDT-Timing-Test/plots/color_wheel.png",
                xref="paper", yref="paper",
                x=1.085, y=0.01,
                sizex=0.35, sizey=0.35,
                xanchor="right", yanchor="bottom"
            )
        )

    fig.update_layout(margin=dict(r=0, l=0, b=0, t=100))

    config = {'displayModeBar': False}
    return fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')