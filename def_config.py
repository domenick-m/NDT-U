#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os.path as osp
#────#
from yacs.config import CfgNode as CN
'''─────────────────────────────── default.py ───────────────────────────────'''
# This file contains the default configs for training the NDT-U model.

config = CN()
config.dirs = CN()
config.data = CN()
config.log = CN()
config.train = CN()
config.model = CN()

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                  DIRS                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.dirs.cfg_path = '/home/dmifsud/Projects/NDT-U/configs/pretrain.yaml'
   # This is the path to the config that will overwrite the default config
config.dirs.sweep_cfg_path = '/home/dmifsud/Projects/NDT-U/configs/sweeps/full_rand_sweep.yaml' 
   # This is the path to the sweep config in wandb's format
config.dirs.trained_mdl_path = '' 
   # The path to the state dict the modle should be initialized with, this will be blank for pretrain
config.dirs.save_dir = '/home/dmifsud/Projects/NDT-U/runs/test' 
   # The directory the model should save state dict to, this will be samba for fine-tuning
config.dirs.dataset_dir = '/home/dmifsud/Projects/NDT-U/data/t5_radial_8'   
   # Path to data directory, should contain sessions.csv, condition_ids.csv, dataset.py, cached_data and cached_pcr
config.dirs.raw_data_dir = '/snel/share/share/data/bg2/t5_cursor'   
   # Path to data directory, should contain sessions.csv

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                  DATA                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.data.bin_size = 10   # ms to bin spikes by
config.data.seq_len = 30   # Chop size in bins
config.data.overlap = 15   # Overlapping bins between chops for training

# Sessions to train on
config.data.sessions = [
   't5.2021.05.05',
   't5.2021.05.17',
   't5.2021.05.19',
   't5.2021.05.24',
   't5.2021.05.26',
   't5.2021.06.02',
   # 't5.2021.06.04',
   # 't5.2021.06.07',
   # 't5.2021.06.23',
   # 't5.2021.06.28',
   # 't5.2021.06.30',
   # 't5.2021.07.07',
   # 't5.2021.07.08',
   # 't5.2021.07.12',
   # 't5.2021.07.14',
   # 't5.2021.07.19',
   # 't5.2021.07.21',
]

config.data.ol_align_field = 'start_time'
config.data.ol_align_range = [0, 2000]
config.data.cl_align_field = 'start_time'
config.data.cl_align_range = [0, 2000]

config.data.lag = 40   # ms to lag behavior by for decoding
config.data.smth_std = 60   # ms std to smooth rates for PCR init
config.data.use_cl = False   # if closed-loop blocks should be used to train model
config.data.rem_xcorr = True   # Whether or not correlated channels should be removed.
config.data.xcorr_thesh = 0.2   # Threshold to remove correlated channels
config.data.pct_heldout = 0.25   # What percentage of channels should be heldout
config.data.heldout_seed = 123456789   # seed for heldout

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                  LOG                                   ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.log.to_csv = False   # 
config.log.to_wandb = False   # Whether or not data is uploaded to wandb
config.log.save_plots = 'both'   # ['local', 'wandb', 'both'] Where evaluation plots should be stored
config.log.wandb_silent = True   # 
config.log.wandb_entity = 'emory-bg2'   # The wandb project the run should be stored in
config.log.wandb_project = 'test'   # The wandb project the run should be stored in

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 TRAIN                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.train.batch_size = 512  # Number of samples to compute loss with
config.train.epochs = 1       # Number of full passes through dataset
config.train.gpu = -1   # seed for training

config.train.early_stopping = False # Whether or not the model stops training due to low co-bps
config.train.es_patience = 500
config.train.es_metric = 'hi_lt_cobps'
config.train.val_interval = 1 # Epochs between running on the validation set
config.train.val_type = 'random' # ['random', 'cross_val']
config.train.pct_val = 0.2   # Percentage of training data in validation set
config.train.n_folds = 5 # Number of folds (K) to use with K-folds cv

config.train.init_lr = 0.005   # The initial learning rate to be used by the optimizer
# config.train.init_lr = 0.005   # The initial learning rate to be used by the optimizer
config.train.optimizer = 'AdamW'   # ['AdamW',] The optimizer to use
config.train.weight_decay = 5.0e-05   # The weight decay value used by AdamW, kind of like L2 Reg but better
config.train.scheduler = 'Cosine'   # ['None', 'Cosine',] The learning rate scheduler
config.train.warmup_steps = 200   # !!!!!!!!! TEST THIS FOR EPOCHS VS STEPS !!!!!    Warmup epcohs used by Cosine scheduler, icreases lr to 1 in this many epochs before it follows cosine decay
config.train.max_grad_norm = 200.0   # The max gradient before value is clipped
config.train.log_eps = 1e-8 # The epsilon to be added to log, should be really small

config.train.mask_max_span = 4 # The max number of timesteps that can be masked in a row randomly 
config.train.ramp_start = 500 # Epoch when the expand prob starts to increase
config.train.ramp_end = 1000 # Epoch when the expand prob remains at mask_max_span

config.train.seed = 123456789   # seed for training
config.train.val_seed = 123456789   # seed for getting validation set

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 MODEL                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.model.factor_dim = 64 # Dimensions that NDT will use after readin / before readout
config.model.n_layers = 4 # The number of EncoderLayers the Encoder should have
config.model.n_heads = 2 # The number of heads used in UndividedMultiheadAttention
config.model.head_dim = 64 # The number of heads used in UndividedMultiheadAttention
config.model.hidden_size = 128 # The size of the linear layers in each EncoderLayer
config.model.context_forward = 1 # How many timesteps in the future can a timestep attend to
config.model.context_backward = 1 # How many timesteps in the past can a timestep attend to

config.model.freeze_model = True # When fine-tuning should only the readout be trained
config.model.rand_readin_init = False # If True, readin is randomly initalized; if False, readin is initalized with PCR
config.model.freeze_readin = True # If rand_readin_init is False, should the readin be frozen

config.model.cat_pos_emb = False
config.model.pos_emb_size = 64 
config.model.scale_input = True

config.model.norm = 'layer' # ['layer', 'scale'] The normalization to be used in the EncoderLayers
config.model.activation = 'relu' # ['relu', 'gelu']
config.model.normal_init = False # do norm tests!!!

config.model.dropout = 0.3 # Overall dropout, used in EncoderLayer
config.model.dropout_rates = 0.3 # Dropout of model output (rates)
config.model.dropout_embedding = 0.3 # Dropout applied after pos_embedding is added
config.model.dropout_attention = 0.3 # Dropout applied to the attention matrix in UndividedMultiheadAttention

config.model.loss_ratio = 0.25 # Percentage of tokens that loss is computed with
config.model.mask_ratio = 0.75 # Percentage of tokens being used to compute the loss are zero masked
config.model.random_ratio = 1.0 # Percentage of unmasked tokens loss is computed with that should be randomized


