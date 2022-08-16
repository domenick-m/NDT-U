import numpy as np
import matplotlib.pyplot as plt
import torch

import sys
#────#
import torch
from nlb_tools.make_tensors import save_to_h5
#────#
from datasets import get_dataloaders
from configs.default_config import get_config_from_file
from setup import set_device, set_seeds, setup_runs_folder

import scipy.signal as signal
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
'''──────────────────────────────── test.py ─────────────────────────────────'''
# This file takes in the path to a .pt file as an argument and saves
# submission.h5 in save_path/test/run_name where run_name is the name of the
# folder that the .pt file is stored in.

def standalone_plot():
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
        exit()
    path = sys.argv[1]
    config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')

    save_path = path[:path.rindex('/')+1]

    if config['train']['seq_len'] > 0:
        rates = torch.load(save_path+'lt_eval_rates.pt')
        spikes = torch.load(save_path+'lt_eval_spikes.pt')
    else:
        rates = torch.load(save_path+'eval_rates.pt')
        spikes = torch.load(save_path+'eval_spikes.pt')

    train_dataloader, val_dataloader = get_dataloaders(config, config['train']['val_type'])
    if config['train']['seq_len'] > 0:
        rates = rates.reshape((
            val_dataloader.dataset.spikes_pre_chop.shape[0],
            rates.shape[0] // val_dataloader.dataset.spikes_pre_chop.shape[0],
            rates.shape[1],
            rates.shape[2]
        ))
        spikes = spikes.reshape((
            val_dataloader.dataset.spikes_pre_chop.shape[0],
            spikes.shape[0] // val_dataloader.dataset.spikes_pre_chop.shape[0],
            spikes.shape[1],
            spikes.shape[2]
        ))
    plot(config['train']['seq_len'] > 0, rates, spikes, save_path)


def plot(chopped, rates, spikes, save_path):
    if chopped:
        rates = torch.cat([rates[:, 0, :-1, :], rates[:, :, -1, :], ], 1)
        spikes = torch.cat([spikes[:, 0, :-1, :], spikes[:, :, -1, :]], 1)

    kern_sd_ms = 40
    kern_sd = int(round(kern_sd_ms / 5))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')

    smooth_spikes = np.apply_along_axis(filt, 1, spikes.cpu().numpy())

    for trial in [1, 5, 10, 15, 20]:
        len_neruons = rates.shape[-1]
        length = int(np.ceil(np.sqrt(len_neruons)))
        fig, axes = plt.subplots(length -1, length, figsize=(25, 25), sharex='col', sharey='row')

        neuron = 0

        for i in range(length):
            for j in range(length):
                if len_neruons <= neuron:
                    break
                axes[i, j].plot(rates[trial, :, neuron].cpu().numpy(), color='tab:red', label='Rates'if neuron==0 else '')
                axes[i, j].plot(smooth_spikes[trial, :, neuron], color='tab:blue', label='Smoothed Spikes' if neuron==0 else '')
                neuron+=1

        fig.legend()
        fig.text(0.5, 0.98, f'Full Trial Model - Trial {trial}', ha='center', fontsize=20)
        # fig.text(0.5, 0.98, f'Sliding Window Model - Trial {trial}', ha='center', fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.97)
        plt.savefig(save_path+f'trial_{trial}.png',facecolor='white', transparent=False)


if __name__ == "__main__":
    standalone_plot()