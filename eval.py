#!/usr/bin/env python3
# Author: Domenick Mifsud
import torch
import wandb
import pandas as pd
import copy
import os.path as osp
import numpy as np
from utils.training_utils import set_seeds, set_device
from utils.config_utils import get_config_from_file
from model import Transformer
import pickle as pkl
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sys
# from utils.data_utils import chop, get_heldin_mask, smooth
from utils.plot.pcs import plot_pcs
from utils.toolkit_utils import load_toolkit_datasets, get_trialized_data
from utils.eval_utils import run_pca, run_decoding


import matplotlib.pyplot as plt

def run_from_path(path):
    run_dir = osp.dirname(path)
    config = get_config_from_file(osp.join(run_dir, 'config.yaml'))

    set_seeds(config)
    set_device(config, {})

    with open(osp.join(run_dir, 'dataset.pkl'), "rb") as dfile:
        dataset = pkl.load(dfile)

    model = Transformer(config, dataset).to(torch.device('cuda:0'))
    model.load_state_dict(torch.load(path))

    return run_evaluation(config, model)

def run_evaluation(config, model):
    print('\nRunning evaluation...')

    # turn off dropout
    model.eval()

    # device to run inference on
    device = torch.device('cuda:0')

    # load each snel_toolkit dataset into dict
    datasets = load_toolkit_datasets(config)

    # make trialized open- and closed-loop data and run inference on it
    trialized_data = get_trialized_data(config, datasets, model)

    run_name = osp.basename(config.dirs.save_dir)

    if wandb.run is None and (config.log.save_plots in ['wandb', 'both'] or config.log.to_wandb):
        wandb.init(project='plots', name=run_name)

    run_decoding(config, trialized_data)  

    trial_len = (config.data.ol_align_range[1] - config.data.ol_align_range[0]) / config.data.bin_size

    factors = []
    for session in config.data.sessions:    
        for ids, trial in trialized_data[session]['ol_trial_data'].groupby([('cond_id', 'n'), 'trial_id']):
            # do not run pca on return trials
            if ids[0] > 0 and trial.factors_smth.shape[0] == trial_len:
                factors.append(trial.factors_smth.to_numpy())

    factors = np.concatenate(factors, 0)
    pca = PCA(n_components=3)
    pca.fit(factors)

    ol_cond_avg, cl_cond_avg, ol_single_trial, cl_single_trial = run_pca(config, trialized_data, pca)

    pca_plots = [
        (ol_cond_avg, 'OL Condition Averaged PCs', 'ol_cond_avg_pcs'),
        (cl_cond_avg, 'CL Condition Averaged PCs', 'cl_cond_avg_pcs'),
        (ol_single_trial, 'OL Single Trial PCs', 'ol_single_tr_pcs'),
        (cl_single_trial, 'CL Single Trial PCs', 'cl_single_tr_pcs'),
    ]

    plot_names, html_strings = [], []

    for plot in pca_plots:
        plot_names.append(plot[1])
        html_strings.append(plot_pcs(*plot[0], plot[1], animate=True))
        
    if config.log.save_plots in ['local', 'both']:
        for plot, string in zip(pca_plots, html_strings):
             with open(osp.join(config.dirs.save_dir, f'{plot[2]}.html'), 'w') as f: 
                f.write(string)

    if config.log.save_plots in ['wandb', 'both']:
            
        for name, string in zip(plot_names, html_strings):
            wandb.log({name : wandb.Html(string, inject=False)})

    if config.log.save_plots in ['wandb', 'both'] or config.log.to_wandb:
        wandb.finish()
        

if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        run_from_path(sys.argv[1])
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')




