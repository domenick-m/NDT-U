import plotly.graph_objects as go
import numpy as np

def plot_true_vs_pred_mvmnt(tr_ids, pred_movement, true_movement):
    print('\nGenerating "true_vs_pred_movement.html"...')

    fig = go.Figure()
    
    ranges = []

    for pred_move, true_move in zip(pred_movement, true_movement):
        fig.add_trace(go.Scatter(visible=True, line=dict(color="#e15759"), x=np.cumsum(pred_move[0]/100), y=np.cumsum(pred_move[1]/100), name="Predicted Behavior"))
        fig.add_trace(go.Scatter(visible=True, line=dict(color="#4e79a7"), x=np.cumsum(true_move[0]/100), y=np.cumsum(true_move[1]/100), name="True Behavior"))


    for tid, trial in trial_data.groupby('trial_id'):
        x = np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)
        y = trial.pred_vel.to_numpy()[29:, 1]/100
        fig.add_trace(go.Scatter(visible=False, line=dict(color="#e15759"), x=np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100), name="NDT Predicted Reach"))
        fig.add_trace(go.Scatter(visible=False, line=dict(color="#4e79a7"), x=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100), name="Smooth Spikes Predicted Reach"))
        fig.add_trace(go.Scatter(visible=False, line=dict(color="#000000"), x=np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100), y=np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100), name="True Reach"))

    ranges = []
    for tid, trial in trial_data.groupby('trial_id'):
        min_x = min(min(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)), min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100)), min(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100)))
        min_y = min(min(np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100)), min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100)), min(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100)))
        max_x = max(max(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]/100)), max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]/100)), max(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]/100)))
        max_y = max(max(np.cumsum(trial.pred_vel.to_numpy()[29:, 1]/100)), max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]/100)), max(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]/100)))
        pad = 0.05

        x_len = (max_x - min_x) * pad
        y_len = (max_y - min_y) * pad

        ranges.append(([ min_x - x_len, max_x + x_len], [min_y - y_len, max_y + y_len]))

    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True

    steps = []
    for i in range(int(len(fig.data)/3)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                #   {"title": "Slider switched to step: " + str(i), 
                {"xaxis" : dict(
                    range=ranges[i][0], 
                    tickmode = 'linear',
                    tick0=0,
                    dtick=10, 
                    zeroline=True, 
                    zerolinewidth=2, 
                    zerolinecolor='slategray',
                    title="Horizontal Movement Distance (mm)", 
                    fixedrange=True 
                ),
                "yaxis" : dict(
                    scaleanchor = "x", 
                    scaleratio = 1, 
                    range=ranges[i][1], 
                    zeroline=True, 
                    zerolinewidth=2, 
                    zerolinecolor='slategray',
                    tickmode = 'linear',
                    tick0=0,
                    dtick=10,
                    title="Vertical Movement Distance (mm)", 
                    fixedrange=True 
                )}],
            label=f'{i}'
        )
        step["args"][0]["visible"][i*3] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i*3+1] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i*3+2] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Trial: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        legend=dict(
            yanchor="bottom",
            y=1.035,
            xanchor="right",
            x=1.00
        ),
        xaxis_title="Horizontal Movement Distance (mm)",
        yaxis_title="Vertical Movement Distance (mm)",
        title="True vs Predicted Movements",
    )

    fig.update_xaxes(
        range=ranges[0][0], 
        tickmode = 'linear',
        tick0=0,
        dtick=10, 
        zeroline=True, 
        zerolinewidth=2, 
        zerolinecolor='slategray', 
        fixedrange=True
    )
    fig.update_yaxes(
        scaleanchor = "x", 
        scaleratio = 1, 
        range=ranges[0][1], 
        zeroline=True, 
        zerolinewidth=2, 
        zerolinecolor='slategray',
        tickmode = 'linear',
        tick0=0,
        dtick=10, 
        fixedrange=True
    )
    layout = go.Layout(
        margin=go.layout.Margin(
            l=65, #left margin
            r=25, #right margin
            b=135, #bottom margin
            t=0  #top margin
        )
    )
    fig.update_layout(layout)
    # config = {'displayModeBar': False}
    # fig.write_html(f"plots/{name}/true_vs_pred_movement.html", config=config)
    # wandb.log({"pred movement plot": wandb.Html(open(f"plots/{name}/true_vs_pred_movement.html"), inject=False)})


    print("Done!\n")

    config = {'displayModeBar': False}
    return fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')
