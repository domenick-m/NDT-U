from inspect import cleandoc
import plotly.graph_objects as go

def plot_rates_vs_spks_indv(
    rates, 
    c_rates, 
    ac_rates, 
    smth_spikes_30, 
    smth_spikes_50, 
    smth_spikes_80,
    heldin
):

    fig = go.Figure()
    x_range=2500

    rates_arr = [] 
    c_rates_arr = [] 
    ac_rates_arr = [] 
    smth_spikes_30_arr = [] 
    smth_spikes_50_arr = [] 
    smth_spikes_80_arr = [] 

    first_idx = 0 if heldin else 98

    rates_arr.append(list(rates[:x_range, first_idx]))
    c_rates_arr.append(list(ac_rates[:x_range, first_idx]))
    ac_rates_arr.append(list(c_rates[:x_range, first_idx]))
    smth_spikes_30_arr.append(list(smth_spikes_30[:x_range, first_idx]))
    smth_spikes_50_arr.append(list(smth_spikes_50[:x_range, first_idx]))
    smth_spikes_80_arr.append(list(smth_spikes_80[:x_range, first_idx]))
    
    fig.add_trace(go.Scatter(y=rates_arr[0], line=dict(color="#e15759"), name="NDT Rates",))
    fig.add_trace(go.Scatter(y=smth_spikes_30_arr[0], line=dict(color="#4e79a7"), name="Smooth Spikes"))
  
    min, max = (1, 98) if heldin else (99, 130)

    for i in range(min, max):
        rates_arr.append(list(rates[:x_range, i]))
        c_rates_arr.append(list(ac_rates[:x_range, i]))
        ac_rates_arr.append(list(c_rates[:x_range, i]))
        smth_spikes_30_arr.append(list(smth_spikes_30[:x_range, i]))
        smth_spikes_50_arr.append(list(smth_spikes_50[:x_range, i]))
        smth_spikes_80_arr.append(list(smth_spikes_80[:x_range, i]))
    
        fig.add_trace(go.Scatter(y=rates_arr[i], visible=False, line=dict(color="#e15759"), name="NDT Rates"))
        fig.add_trace(go.Scatter(y=smth_spikes_30_arr[i], visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(), rangeslider=dict(visible=True)
        )
    )

    rate_idx = [i for i in range(0, 196, 2)]
    smth_idx = [i for i in range(1, 196, 2)]

    spks_smth_buttons = [
        {
            'method': 'update',
            'label': f'30ms',
            'visible': True,
            'args': [{'y': smth_spikes_30_arr}, {}, smth_idx] 
        },
        {
            'method': 'update',
            'label': f'50ms',
            'visible': True,
            'args': [{'y': smth_spikes_50_arr}, {}, smth_idx] 
        },
        {
            'method': 'update',
            'label': f'80ms',
            'visible': True,
            'args': [{'y': smth_spikes_80_arr}, {}, smth_idx] 
        },
    ]

    rates_smth_buttons = [
        {
            'method': 'update',
            'label': f'None',
            'visible': True,
            'args': [{'y': rates_arr}, {}, rate_idx],
        },
        {
            'method': 'update',
            'label': f'Acausal (60ms std)',
            'visible': True,
            'args': [{'y': ac_rates_arr}, {}, rate_idx],
        },
        {
            'method': 'update',
            'label': f'Causal (60ms std)',
            'visible': True,
            'args': [{'y': c_rates_arr}, {}, rate_idx],
        },
    ]

    ch_sel_buttons = []
    min, max = (0, 98) if heldin else (98, 130)

    for i in range(min, max):
        vis_list = [False for i in range(196 if heldin else 64)]
        vis_list[i*2 if heldin else (i-98)*2] = True
        vis_list[i*2+1 if heldin else (i-98)*2+1] = True
        ch_sel_buttons.append(dict(
            method='restyle',
            label=f'ch {i+1}',
            visible=True,
            args=[{'visible':vis_list}]
        ))
            
    # specify updatemenu        
    um = [
        {
            'buttons':ch_sel_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b":20},
            'showactive':True,
            'x':0.5,
            'xanchor':"center",
            'y':1.00,
            'yanchor':"bottom" 
        },
        {
            'buttons':spks_smth_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b":20},
            'showactive':True,
            'x':0.25,
            'xanchor':"center",
            'y':1.00,
            'yanchor':"bottom"
        },
        {
            'buttons':rates_smth_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b":20},
            'showactive':True,
            'x':0.75,
            'xanchor':"center",
            'y':1.00,
            'yanchor':"bottom"
        }
    ]
    fig.update_layout(updatemenus=um)

    fig['layout']['xaxis'].update(range=['0', '301'])

    layout = go.Layout(
        margin=go.layout.Margin(
            l=60, #left margin
            r=0, #right margin
            b=0, #bottom margin
            t=100  #top margin
        )
    )
    fig.update_layout(layout)

    fig.update_xaxes(
        ticktext=[f'{int(i * 10)}ms' for i in range(0, x_range, 25)],
        tickvals=[i for i in range(0, x_range, 25)],
    )

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1.00,
        ),
        yaxis_title="Spikes per Second",
        title={
            'text': f'Rates vs Smoothed Spikes - {"Heldin" if heldin else "Heldout"} Channels',
            'y':1.2,
            'x':0.1,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        annotations=[
            dict(
                text="Trace type:", 
                showarrow=False,
                x=0, 
                y=1.08, 
                yref="paper", 
                align="left"
            )
        ]
    )

    config = {'displayModeBar': False}
    return fig.to_html(config=config)
