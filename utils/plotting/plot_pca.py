from matplotlib import cm, colors
import numpy as np
import plotly.graph_objects as go
import copy


def plot_pca(
    trialized_pcs, 
    conditions, 
    title
):
    # decrease storage footprint as much as possible
    trialized_pcs = [trial.astype(np.float16) for trial in trialized_pcs]
    
    norm = colors.Normalize(vmin=-180, vmax=180, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
    mapper.set_array([])

    fig = go.Figure()

    for trial, cond_id in zip(trialized_pcs, conditions):
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
    return fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')
    