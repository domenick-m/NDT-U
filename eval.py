import torch
from utils.t5_utils import load_toolkit_datasets
import pandas as pd
import copy
import os.path as osp
import numpy as np
from utils.training_utils import set_seeds, set_device
from utils.config_utils import get_config_from_file
from transformer_ import Transformer
import pickle as pkl
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sys
# from utils.data_utils import chop, get_heldin_mask, smooth
from utils.plotting.plot_pca import plot_pca
from utils.t5_utils import get_trialized_data
from utils.eval_utils import run_pca


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
    # turn off dropout
    model.eval()

    # device to run inference on
    device = torch.device('cuda:0')

    # load each snel_toolkit dataset into dict
    datasets = load_toolkit_datasets(config)

    # make trialized open- and closed-loop data and run inference on it
    trialized_data = get_trialized_data(config, datasets, model)
    
    factors = []
    for session in config.data.sessions:    
        for ids, trial in trialized_data[session]['ol_trial_data'].groupby(['condition', 'trial_id']):
            # do not run pca on return trials
            if ids[0] != 0:
                factors.append(trial.factors_smth.to_numpy())

    factors = np.concatenate(factors, 0)
    pca = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
    pca.fit(factors)

    ol_cond_avg, ol_single_trial, cl_single_trial = run_pca(config, trialized_data, pca)

    html_string = plot_pca(*ol_cond_avg, 'OL Condition Averaged')
    with open(f'{config.dirs.save_dir}/ol_cond_avg_.html', 'w') as f: 
        f.write(html_string)
        
    html_string = plot_pca(*ol_single_trial, 'OL Single Trials')
    with open(f'{config.dirs.save_dir}/ol_sing_trial.html', 'w') as f: 
        f.write(html_string)

    html_string = plot_pca(*cl_single_trial, 'CL Single Trials')
    with open(f'{config.dirs.save_dir}/cl_sing_trial.html', 'w') as f: 
        f.write(html_string)

if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        run_from_path(sys.argv[1])
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')




