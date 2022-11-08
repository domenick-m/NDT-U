#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
from tkinter import E
from yacs.config import CfgNode as CN
'''─────────────────────────────── default.py ───────────────────────────────'''
# This file contains the default configs for training the NDT-U model.

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 SETUP                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config = CN()
config.setup = CN()
config.setup.dataset = 'mc_rtt' # ['mc_rtt', 't5_cursor']
config.setup.seed = 80891122 # Seed for initializing model, randomization of dataloader, etc..

config.setup.gpu = -1 # Index of GPU to use, if -1 then Auto-select
config.setup.ag_gpus = [-1] # GPUs for agents to use, if -1 add agent to all GPUs

config.setup.cfg_path = '' # 'configs/example_config.yaml' - The config to overwirte def_config with
config.setup.data_dir = 'data/' # Where each datasets .h5 file is stored
config.setup.save_dir = 'runs/' # Where the train and test output data should be stored

config.setup.save_model = True # If True, save the best (co-bps on val set) model and the model after fully training
config.setup.comp_metric = 'lt_nll' # ['lt_nll', 'lt_co_bps']
config.setup.log_eps = 1e-8 # The epsilon to be added to log, should be really small

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                  DATA                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.data = CN()
config.data.dir = '/home/dmifsud/Projects/NDT-U/data'   # Path to data directory, should contain sessions.csv

# Sessions used for training 
config.data.pretrain_sessions = [
   't5.2021.05.05',
   # 't5.2021.05.17',
   # 't5.2021.05.19',
   # 't5.2021.05.24',
   # 't5.2021.05.26',
   # 't5.2021.06.02',
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
# Sessions used for testing 
config.data.finetune_sessions = [
   't5.2021.05.05'
] 

config.data.bin_size = 10   # ms to bin spikes by
config.data.seq_len = 30   # Chop size in bins
config.data.overlap = 25   # Overlapping bins between chops

config.data.lag = 40   # ms to lag behavior by 
config.data.smth_std = 60   # ms std to smooth rates by when decoding

config.data.center_trials = False   # Should return (center target) trials be used in test evaluation
config.data.trial_len = 2000   # ms after 'start_time' that a trial should span for test evaluation

config.data.rem_xcorr = True   # Whether or not correlated channels should be removed.
config.data.heldout_pct = 0.25   # What percentage of channels should be heldout



'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 TRAIN                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.train = CN()
config.train.batch_size = 1024   # Number of samples to compute loss with
config.train.epochs = 10000       # Number of full passes through dataset

config.train.val_interval = 20 # Epochs between running on the validation set
config.train.val_type = 'random' # ['random', 'last', 'cross_val']
config.train.n_folds = 5 # Number of folds (K) to use with K-folds cv

config.train.sweep_enabled = False # Whether or not wandb hyperparameter sweep is enabled
config.train.sweep_type = 'random' # ['grid', 'random'] type search used for HP sweep
config.train.sweep_epochs = 99999 # Number of models that should be trained if sweep_type is random

config.train.early_stopping = False # Whether or not the model stops training due to low co-bps
config.train.es_patience = 500

config.train.lt_loss_only = False
config.train.always_lt_loss = False
config.train.init_lr = 0.005   # The initial learning rate to be used by the optimizer
config.train.max_grad_norm = 200.0   # The max gradient before value is clipped
config.train.optimizer = 'AdamW'   # ['AdamW',] The optimizer to use
config.train.weight_decay = 5.0e-07   # The weight decay value used by AdamW, kind of like L2 Reg but better
config.train.scheduler = 'Cosine'   # ['None', 'Cosine',] The learning rate scheduler
config.train.warmup_steps = 500   # !!!!!!!!! TEST THIS FOR EPOCHS VS STEPS !!!!!    Warmup epcohs used by Cosine scheduler, icreases lr to 1 in this many epochs before it follows cosine decay

config.train.mask_max_span = 3 # The max number of timesteps that can be masked in a row randomly 
config.train.ramp_start = 8000 # Epoch when the expand prob starts to increase
config.train.ramp_end = 12000 # Epoch when the expand prob remains at mask_max_span

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 MODEL                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.model = CN()
config.model.n_heads = 4 # The number of heads used in UndividedMultiheadAttention
config.model.undivided_attn = True 
config.model.n_layers = 4 # The number of EncoderLayers the Encoder should have
config.model.factor_dim = 128 # Dimensions that NDT will use after readin / before readout
config.model.hidden_size = 256 # The size of the linear layers in each EncoderLayer

config.model.freeze_readin = False # The size of the linear layers in each EncoderLayer
config.model.rand_readin_init = False # The size of the linear layers in each EncoderLayer

config.model.cat_pos_emb = False
config.model.pos_emb_size = 32 
config.model.scale_input = True

config.model.norm = 'layer' # ['layer', 'scale'] The normalization to be used in the EncoderLayers
config.model.gnorm_groups = 10 # ['layer', 'scale'] The normalization to be used in the EncoderLayers
config.model.activation = 'relu' # ['relu', 'gelu']

config.model.dropout = 0.7 # Overall dropout, used in EncoderLayer
config.model.dropout_rates = 0.3 # Dropout of model output (rates)
config.model.dropout_embedding = 0.7 # Dropout applied after pos_embedding is added
config.model.dropout_attention = 0.6 # Dropout applied to the attention matrix in UndividedMultiheadAttention

config.model.normal_init = False
config.model.initrange = 0.1 # The range that should be used on the normal init of the decoder

config.model.loss_ratio = 0.25 # Percentage of tokens that loss is computed with
config.model.mask_ratio = 0.75 # Percentage of tokens being used to compute the loss are zero masked
config.model.random_ratio = 1.0 # Percentage of unmasked tokens loss is computed with that should be randomized

config.model.context_forward = 20 # How many timesteps in the future can a timestep attend to
config.model.context_backward = 30 # How many timesteps in the past can a timestep attend to

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 WANDB                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.wandb = CN()
config.wandb.log = True                  # Whether or not data is uploaded to wandb
config.wandb.entity = 'emory-bg2'        # The wandb project the run should be stored in
config.wandb.project = 'High Dropout Sweep' # The wandb project the run should be stored in
config.wandb.sweep_name = 'my-sweep'     # The name of the sweep if train.sweep_enabled is True
config.wandb.log_local = True            # If wandb.log is False should logs (what would be uploaded to wandb) be saved locally to train/runs/run_name/report_log.txt'
config.wandb.silent = True               # ['true', 'false'] If 'true' wandb does not print anything
'''
                    ►──────────── WANDB.SWEEP ────────────◄
'''
config.wandb.sweep = CN()
config.wandb.sweep.setup = CN()

config.wandb.sweep.data = CN()
# config.wandb.sweep.data.rem_xcorr = [True, False]
config.wandb.sweep.train = CN()
# config.wandb.sweep.train.batch_size = [1024, 2048]
# config.wandb.sweep.train.e_batch_size = [128, 256, 512, 1024, 2048, 4096]
# config.wandb.sweep.train.warmup_steps = [100, 500, 1000, 2500, 5000]
# config.wandb.sweep.train.weight_decay = [5.0e-03, 5.0e-05, 5.0e-07]
# config.wandb.sweep.train.init_lr = [0.1, 0.01, 0.001]
# config.wandb.sweep.train.always_lt_loss = [True, False]

config.wandb.sweep.model = CN()
# config.wandb.sweep.model.undivided_attn = [True, False]
# # config.wandb.sweep.model.freeze_readin = [True, False]
# # config.wandb.sweep.model.rand_readin_init = [True, False]
config.wandb.sweep.model.n_heads = [2, 3, 4, 5]
config.wandb.sweep.model.mask_ratio = [0.75, 0.95]
# config.wandb.sweep.model.cat_pos_emb = [True, False]
# config.wandb.sweep.model.pos_emb_size = [16, 32, 64]
# config.wandb.sweep.model.scale_input = [True, False]

config.wandb.sweep.model.n_layers = [4, 5, 6, 7]
# config.wandb.sweep.model.hidden_size = [256]
# # config.wandb.sweep.model.factor_dim = [16, 32, 64, 128, 256]
config.wandb.sweep.model.dropout = [0.6, 0.65, 0.7, 0.75, 0.8]
config.wandb.sweep.model.dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
config.wandb.sweep.model.dropout_embedding = [0.6, 0.65, 0.7, 0.75, 0.8]
config.wandb.sweep.model.dropout_attention = [0.6, 0.65, 0.7, 0.75, 0.8]
config.wandb.sweep.model.context_forward = [10, 15, 20, 25, 30]
config.wandb.sweep.model.context_backward = [20, 25, 30]
# config.wandb.sweep.model.normal_init = [True, False]
