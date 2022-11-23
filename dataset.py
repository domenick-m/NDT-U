#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import copy
import random
import logging
import requests
import pandas as pd
import os.path as osp
#────#
import h5py
import torch
import numpy as np
import scipy.signal as signal
from tqdm.auto import tqdm
from torch.utils import data
from nlb_tools import make_tensors
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import KFold
from nlb_tools.make_tensors import (
    save_to_h5,
    combine_h5,
    h5_to_dict,
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors)
from utils_f import set_seeds
from data.t5_dataset import T5CursorDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler



import warnings; warnings.simplefilter('ignore')

class Dataset(data.Dataset):
    def __init__(self, config, chopped_spikes, session_names):
        '''init Dataset

        Args:
            config (dict): A config object.
            filename (str): path to the dataset.h5 file.
            mode (str): ['test', 'train' , 'val', 'trainval'] The dataset type.
        '''
        super().__init__()
        device = torch.device('cuda:0')

        self.session_names = session_names
        self.n_samples = chopped_spikes.shape[0]
        self.n_channels = chopped_spikes.shape[-1]

        self.n_heldout = int(config.data.pct_heldout * self.n_channels)
        self.n_heldin = self.n_channels - self.n_heldout

        # If the percentage of heldout channels is above 0, remove heldout channels
        # if self.n_heldout > 0.0:
        self.has_heldout = self.n_heldout > 0

        np.random.seed(config.data.heldout_seed)
        heldout_channels = np.random.choice(self.n_channels, self.n_heldout, replace=False)
        heldin_channels = torch.ones(self.n_channels, dtype=bool)
        heldin_channels[heldout_channels] = False

        # self.heldin_spikes = chopped_spikes[:, :, heldin_channels]
        # self.heldout_spikes = chopped_spikes[:, :, heldout_channels]

        # print(self.heldin_spikes.shape)
        # print(self.heldout_spikes.shape)
        self.heldin_spikes = chopped_spikes[..., heldin_channels].to(device)
        self.heldout_spikes = chopped_spikes[..., heldout_channels].to(device)
        # else:
        #     self.has_heldout = False
        #     self.heldin_spikes = chopped_spikes.to(device)

        self.max_train_spks = self.heldin_spikes.max().item() + 3

    def __len__(self):
        '''
        Returns:
            int: The number of samples or trials.
        '''
        return self.n_samples

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index into the batches.
        Returns:
            spikes_heldin (np.ndarray): Spikes held-in.
            spikes_heldout (np.ndarray): Spikes held-out.
            spikes_all_fp (np.ndarray): Spikes forward pass.
        '''
        heldout = self.heldout_spikes[index] if self.has_heldout else []
        return self.heldin_spikes[index], heldout, self.session_names[index]

    def get_dataloader(self, generator, shuffle=True):
        '''
        Args:
        Returns:
        '''
        return data.DataLoader(
            self,
            batch_size=self.config.train.batch_size,
            generator=generator,
            shuffle=shuffle
        )

