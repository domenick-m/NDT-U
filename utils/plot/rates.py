from inspect import cleandoc
import plotly.graph_objects as go

def plot_rates_vs_spks_all(channels, train_rates, smth_spikes):
    def rates_string(neuron):
        array_string = 'y: ['
        for i in train_rates[:1000,neuron]:
            array_string += str(i.item())+','
        array_string += '],'
        return array_string

    def ss_string(neuron):
        array_string = 'y: ['
        for i in smth_spikes[:1000,neuron]:
            array_string += str(i.item())+','
        array_string += '],'
        return array_string

    # with open(f"plots/{name}/ho_all_spk_vs_rates.html", "w") as f:
    html_string = cleandoc(
        # f.write(cleandoc(
        '''<!DOCTYPE html><html lang="en" >
        <head><meta charset="UTF-8">
            <title>NDT Heldout Rates</title>
        </head>
        <body>
        <!-- partial:index.partial.html --><div id="legend" style="height: 50px"></div>
        <div style="height:450px; overflow-y: auto">
            <div id="plot" style="height:8000px"></div>
        </div>
        <div id="xaxis" style="height: 60px"></div>
        <!-- partial --><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.3.1/plotly.min.js"></script>
        <script>'''
    )
    names = []
    if channels == 'heldin':
        min, max = 0, 98
    else:
        min, max = 98, 130

    for i in range(min, max):
        names.append(f'trace{i+1}')
        names.append(f'trace{i+1}r')
        str_to_write = f'var trace{i+1} = {{'
        str_to_write += ss_string(i)
        str_to_write += cleandoc(
            f"""marker: {{color: '#4e79a7'}},
            name: 'Smoothed Spikes',
            yaxis: 'y{i+1 if channels == 'heldin' else i-97}',
            type: 'line',"""
        )
        if i != 0:
            str_to_write += "showlegend: false,"
        str_to_write += f'}};\nvar trace{i+1}r = {{'
        str_to_write += rates_string(i)
        str_to_write += cleandoc(
            f"""marker: {{color: '#e15759'}}, 
            name: 'NDT Rates', 
            yaxis: 'y{i+1 if channels == 'heldin' else i-97}', 
            type: 'line',"""
        )
        if i != 0:
            str_to_write += "showlegend: false,"
        str_to_write +='};\n'
        html_string += str_to_write
    names_str = 'data = ['
    for i in names:
        names_str += f"{i}, "
    names_str += ']'
    html_string += names_str+f'\n'
    html_string += f'var config = {{responsive: true, displayModeBar: false}};'
    
    html_string += cleandoc(
        f'''var layout_hide_ticks = {{
            xaxis: {{visible: false, fixedrange: true}},
            grid: {{rows: {98 if channels == 'heldin' else 32}, columns: 1}},
            '''
    )
    axis_labels = f"\nyaxis: {{title: {{text: 'ch {1 if channels == 'heldin' else 99}',}}, showticklabels: false, fixedrange: true}},\n"
    
    if channels == 'heldin':
        min, max = 2, 99
    else:
        min, max = 100 , 131

    for i in range(min,max):
        axis_labels += f"yaxis{i if channels == 'heldin' else i-98}: {{title: {{text: 'ch {i}',}}, showticklabels: false, fixedrange: true}},\n"
    html_string += axis_labels
    html_string += 'margin: { l: 25, t: 45, b: 0 , r: 25},showlegend: false,}; '

    html_string += f'var layout_show_ticks = {{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: {98 if channels == "heldin" else 32}, columns: 1}},'
    axis_labels = f"\nyaxis: {{title: {{text: 'ch {1 if channels == 'heldin' else 99}',}}, showticklabels: true, fixedrange: true}},\n"

    for i in range(min,max):
        axis_labels += f"yaxis{i if channels == 'heldin' else i-98}: {{title: {{text: 'ch {i}',}}, showticklabels: true, fixedrange: true}},\n"
    html_string += axis_labels
    html_string += 'margin: { l: 60, t: 45, b: 0 , r: 25},showlegend: false,}; '

    html_string += cleandoc(
        '''
        var updatemenus=[
            {
                buttons: [
                    {
                        args: [{}, {...layout_hide_ticks}], 
                        label: "Hide Y Axis Labels", 
                        method: "update"
                    }, 
                    {
                        args: [{}, {...layout_show_ticks}], 
                        label:"Show Y Axis Labels (spikes/sec)", 
                        method:"update"
                    }
                ], 
                direction: "down", 
                pad: {"r": 0, "t": 0, "b":0, "l": 0}, 
                showactive: true, 
                type: "dropdown", 
                x: 0, 
                xanchor: "left", 
                y: 1.0175, 
                yanchor: "top",
            }, 
            {
                buttons: [
                    {
                        args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 1000]}}], 
                        label: "10s", 
                        method: "update"
                    }, 
                    {
                        args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 500]}}], 
                        label:"5s", 
                        method:"update"
                    }, 
                    {
                        args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 250]}}], 
                        label:"2.5s", 
                        method:"update"
                    }, 
                        {args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 100]}}], 
                        label:"1s", 
                        method:"update"
                    }, 
                    {
                        args: [{}, {xaxis: {visible: false, fixedrange: true, range: [0, 50]}}], 
                        label:"500ms", 
                        method:"update"
                    },
                ], 
                direction: "down", 
                pad: {"r": 0, "t": 0, "b":0, "l": 0}, 
                showactive: true, 
                type: "dropdown", 
                x: 0.5, 
                xanchor: "center", 
                y: 1.0175, 
                yanchor: "top",
            },
        ]; 
        layout_hide_ticks["updatemenus"] = updatemenus; 
        var config = {responsive: true, displayModeBar: false}; 
        Plotly.react("plot",data,layout_hide_ticks,config);
        let bottomTraces = [{mode: "scatter" }]; 
        var graphDiv = document.getElementById("plot"); 
        var axisDiv = document.getElementById("xaxis"); 
        let bottomLayout = {
            yaxis: { tickmode: "array", tickvals: [], fixedrange: true },
            xaxis: {
                tickVals: [0, 250, 500, 750, 999], 
                tickText: ["0s", "2.5s", "5s", "7.5s", "10s"], 
                tickmode: "array",
                range: graphDiv.layout.xaxis.range,
                domain: [0.0, 1.0],
                fixedrange: true
            },
            margin: { l: 25, t: 0 , r: 40},
        }; 
        Plotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false, responsive: true }); 
        data = [
            {
                y: [null],
                name: "Smooth Spikes",
                mode: "lines",
                marker: {color: "#4e79a7"},
            },
            {
                y: [null],
                name: "NDT Rates",
                mode: "lines",
                marker: {color: "#e15759"},
            }
        ]; 
        let newLayout = {
            title: {
                text: "Rates vs Smoothed Spikes - 
        '''
    ) + (
        'Heldin' if channels == 'heldin' else 'Heldout'
    ) + cleandoc(
        '''
         Channels", 
                y:0.5, 
                x:0.025
            },
            yaxis: { visible: false},
            xaxis: { visible: false},
            margin: { l: 0, t: 0, b: 0, r: 0 },
            showlegend: true,
        }; 
        Plotly.react("legend", data, newLayout, { displayModeBar: false, responsive: true }); 
        var range = 1000; 
        graphDiv.on("plotly_afterplot", function(){ 
            var tickVals = [0, 250, 500, 750, 999]; 
            var tickText = ["0s", "2.5s", "5s", "7.5s", "10s"]; 
            if (graphDiv.layout.xaxis.range[1] == 1000) { 
                var tickVals = [0, 250, 500, 750, 999]; 
                var tickText = ["0s", "2.5s", "5s", "7.5s", "10s"]; range = 1000;
            } else if (graphDiv.layout.xaxis.range[1] == 500) {
                var tickVals = [0, 125, 250, 375, 500];
                var tickText = ["0s", "1.25s", "2.5s", "3.25s", "5s"]; 
                range = 500;
            } else if (graphDiv.layout.xaxis.range[1] == 250) {
                var tickVals = [0, 62.5, 125, 187.5, 250]; 
                var tickText = ["0s", "0.625s", "1.25s", "1.875s", "2.5s"]; 
                range = 250;
            } else if (graphDiv.layout.xaxis.range[1] == 100) {
                var tickVals = [0, 25, 50, 75, 100]; 
                var tickText = ["0s", "1.25s", "2.5s", "3.25s", "1s"]; 
                range = 100;
            } else if (graphDiv.layout.xaxis.range[1] == 50) {
                var tickVals = [0, 12.5, 25, 37.5, 50]; 
                var tickText = ["0ms", "125ms", "250ms", "325ms", "500ms"]; 
                range = 50;
            };
            if (range != axisDiv.layout.xaxis.range[1]) {
                if (graphDiv.layout.yaxis.showticklabels) { 
                    Plotly.update(axisDiv, bottomTraces, {
                        yaxis: { tickmode: "array", tickvals: [], fixedrange: true },
                        xaxis: {   
                            tickmode: "array",
                            tickvals: tickVals,
                            ticktext: tickText,
                            range: [0, range],
                            domain: [0.0, 1.0],
                            fixedrange: true
                        },
                        margin: { l: 60, t: 0 , r: 40},
                    });
                } else { 
                    Plotly.update(axisDiv, bottomTraces, {
                        yaxis: { tickmode: "array", tickvals: [], fixedrange: true },
                        xaxis: {
                            tickmode: "array",
                            tickvals: tickVals,
                            ticktext: tickText,
                            range: [0, range],
                            domain: [0.0, 1.0],
                            fixedrange: true
                        },
                        margin: { l: 25, t: 0 , r: 40},
                    });
                }
            } else if (graphDiv.layout.xaxis.range[1] == 999) { 
                Plotly.update(graphDiv, bottomTraces, {
                    xaxis: {visible: false, fixedrange: true, range: axisDiv.layout.xaxis.range}}); 
                    range = axisDiv.layout.xaxis.range[1]; 
                    if (graphDiv.layout.yaxis.showticklabels) {
                        Plotly.update(axisDiv, bottomTraces, {
                            yaxis: { tickmode: "array", tickvals: [], fixedrange: true },
                            xaxis: {
                                tickmode: "array",
                                tickvals: tickVals,
                                ticktext: tickText,
                                range: [0, range],
                                domain: [0.0, 1.0],
                                fixedrange: true
                            },
                            margin: { l: 60, t: 0 , r: 40},
                        });
                    } else {
                        Plotly.update(axisDiv, bottomTraces,{
                            yaxis: { tickmode: "array", tickvals: [], fixedrange: true },
                            xaxis: {
                                tickmode: "array",
                                tickvals: tickVals,
                                ticktext: tickText,
                                range: [0, range],
                                domain: [0.0, 1.0],
                                fixedrange: true
                            },
                            margin: { l: 25, t: 0 , r: 40},
                        });
                        }
                    }
                });
        </script></body></html>
        '''
    )
    return html_string

    # wandb.log({f"Spikes vs Rates {'Heldin' if channels == 'heldin' else 'Heldout'}": wandb.Html(html_string, inject=False)})

import numpy as np
def plot_rates_vs_spks_indv(
    rates, 
    c_rates, 
    ac_rates, 
    smth_spikes_30, 
    smth_spikes_50, 
    smth_spikes_80,
    heldin
):
    rates = rates.astype(np.float16)
    c_rates = c_rates.astype(np.float16)
    ac_rates = ac_rates.astype(np.float16)
    smth_spikes_30 = smth_spikes_30.astype(np.float16)
    smth_spikes_50 = smth_spikes_50.astype(np.float16)
    smth_spikes_80 = smth_spikes_80.astype(np.float16)

    fig = go.Figure()
    x_range=1000

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
  
    min, max = (1, 25) if heldin else (99, 130)
    # min, max = (1, 98) if heldin else (99, 130)

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

    rate_idx = [i for i in range(0, 50, 2)]
    smth_idx = [i for i in range(1, 50, 2)]

    # rate_idx = [i for i in range(0, 196, 2)]
    # smth_idx = [i for i in range(1, 196, 2)]

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
    min, max = (0, 25) if heldin else (98, 130)
    # min, max = (0, 98) if heldin else (98, 130)

    for i in range(min, max):
        vis_list = [False for i in range(50 if heldin else 64)]
        # vis_list = [False for i in range(196 if heldin else 64)]
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
            'pad': {"r": 0, "t": 0, "b":0, "l":0},
            'showactive':True,
            'x':0.075,
            'xanchor':"left",
            'y':1.10,
            'yanchor':"bottom" 
        },
        {
            'buttons':spks_smth_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b":0, "l":0},
            'showactive':True,
            'x':0.5,
            'xanchor':"center",
            'y':1.10,
            'yanchor':"bottom"
        },
        {
            'buttons':rates_smth_buttons, 
            'direction': 'down',
            'pad': {"r": 0, "t": 0, "b":0, "l":0},
            'showactive':True,
            'x':1.0,
            'xanchor':"right",
            'y':1.10,
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
            y=1.5,
            xanchor="right",
            x=1.00,
        ),
        yaxis_title="Spikes per Second",
        title={
            'text': f'Rates vs Smoothed Spikes - {"Heldin" if heldin else "Heldout"} Channels',
            'y':0.97,
            'x':0.1,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        annotations=[
            dict(
                text="Channel:", 
                showarrow=False,
                x=0.0, 
                y=1.2, 
                yref="paper", 
                xref="paper",
                xanchor="left", 
                align="left",
            ),
            dict(
                text="Spike Smoothing Std Dev:", 
                showarrow=False,
                x=0.3, 
                y=1.2, 
                yref="paper", 
                xref="paper", 
                xanchor="left", 
                align="left"
            ),
            dict(
                text="Rates Smoothing:", 
                showarrow=False,
                x=0.8, 
                y=1.2, 
                yref="paper", 
                xref="paper",
                xanchor="right",  
                align="left"
            )
        ]
    )

    config = {'displayModeBar': False}
    return fig.to_html(config=config, full_html=False, include_plotlyjs='cdn')
