#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
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
config.setup.dataset = 'mc_rtt_cont_24' # ['area2_bump', 'dmfc_rsg', 'mc_maze', 'mc_maze_small', 'mc_maze_medium', 'mc_maze_large', 'mc_rtt']
config.setup.seed = 100 # Seed for initializing model, randomization of dataloader, etc..
config.setup.subset_seed = 404 # Seed for taking a random validation subset

config.setup.gpu_idx = -1 # Index of GPU to use, if -1 then Auto-select
config.setup.save_model = True # If True, save the best (co-bps on val set) model and the model after fully training
config.setup.save_min_bps = -1000.0 # Best model will not be saved until it reaches this point, set high to avoid saving too often
config.setup.log_eps = 1e-7 # The epsilon to be added to log, should be really small

config.setup.config_dir = 'configs/' # Where the default_config.py and dataset_configs/ are stored
config.setup.data_dir = 'data/' # Where each datasets .h5 file is stored
config.setup.runs_dir = 'runs/' # Where the train and test output data should be stored
'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 TRAIN                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.train = CN()


config.train.batch_size = 256 # Number of samples to compute loss with
config.train.seq_len = 0 # 0 is full trial, above 0 is sliding window (trials are chopped to that value)
config.train.overlap = 45 # 0 is full trial, above 0 is sliding window (trials are chopped to that value)
config.train.epochs = 15000 # Number of full passes through dataset

config.train.val_interval = 10 # Epochs between running on the validation set
config.train.val_type = 'original' # ['original', 'random', 'none'] Original is the nlb given validation set, random is a random subset of the combined train and validation set, none is no validation set
config.train.val_ratio = 0.05 # Percentage of the combined training and validation set to use as the validation set when using random val_type

config.train.sweep_enabled = False # Whether or not wandb hyperparameter sweep is enabled, if False running train.py will train a single model
config.train.sweep_type = 'random' # ['grid', 'random'] Which wandb sweep type to use when sweep is enabled, grid is every combination of the sweep values, and random is random combinations of the sweep values
config.train.sweep_epochs = 9999 # Number of models that should be trained if sweep_type is random
config.train.early_stopping = True # Whether or not the model stops training due to low co-bps
config.train.es_min_bps= 0.0 # The point at which a model will be early stopped if it's co-bps score falls below this
config.train.es_chk_pnt= 0.5 # When should the model start checking if it should early stop, 0.5 = halfway through the total epochs it will starting checking if co-bps falls below es_min_bps

config.train.init_lr = 0.01 # The initial learning rate to be used by the optimizer
config.train.optimizer = 'AdamW' # ['AdamW',] The optimizer to use, other may be added in setup.py
config.train.scheduler = 'Cosine' # ['None', 'Cosine',] The scheduler to use on the learning rate, other may be added in setup.py
config.train.warmup_steps = 1500 # Warmup steps used by Cosine scheduler, icreases lr to 1 in this many steps before it follows cosine decay

config.train.normal_init = True # The maximum value a gradient can have before it is clipped, avoids exploding gradient
config.train.add_one_random = False # The maximum value a gradient can have before it is clipped, avoids exploding gradient

config.train.max_grad_norm = 200.0 # The maximum value a gradient can have before it is clipped, avoids exploding gradient
config.train.weight_decay = 1.000e-7 # The weight decay value used by AdamW, kind of like L2 Reg but better

config.train.mask_max_span = 6 # The max number of timesteps that can be masked in a row
config.train.ramp_start = 1000 # Epoch when the number of timesteps being maksed in a row starts to increase
config.train.ramp_end = 10000 # Epoch when the number of timesteps being maksed in a row stops increasing and stays at mask_max_span
'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 MODEL                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.model = CN()
config.model.n_heads = 2 # The number of heads used in UndividedMultiheadAttention
config.model.n_layers = 4 # The number of EncoderLayers the Encoder should have
config.model.hidden_size = 128 # The size of the linear layers in each EncoderLayer

config.model.emb_size = 32

config.model.dropout = 0.4 # Overall dropout, used in EncoderLayer
config.model.dropout_rates = 0.5 # Dropout of model output (rates)
config.model.dropout_embedding = 0.5 # Dropout applied after pos_embedding is added
config.model.dropout_attention = 0.7 # Dropout applied to the attention matrix in UndividedMultiheadAttention

config.model.loss_ratio = 0.2 # Percentage of tokens that loss is computed with
config.model.mask_ratio = 0.75 # Percentage of tokens being used to compute the loss are zero masked
config.model.random_ratio = 1.0 # Percentage of tokens being used to compute the loss (that are not zero masked) that should be randomized

config.model.undivided_attn = False # Percentage of tokens being used to compute the loss (that are not zero masked) that should be randomized

config.model.norm = "layer" # ['layer', 'scale'] The normalization to be used in the EncoderLayers
config.model.activation = "relu" # ['relu', 'gelu']
config.model.max_spike_count = 20 # Max number of spikes allowed, any count above is clipped to this

config.model.xavier = False # Whether or not xaiver init should be used, if False use the init from T-fixup
config.model.initrange = 0.01 # The range that should be used on the normal init of the decoder

config.model.context_forward = 3 # How many timesteps in the future can a timestep attend to
config.model.context_backward = 7 # How many timesteps in the past can a timestep attend to
'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                                 WANDB                                  ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
config.wandb = CN()
config.wandb.log = True # Whether or not data is uploaded to wandb
config.wandb.log_freq = 250 # Epochs between each gradient log of the model by wandb
config.wandb.log_local = False # If wandb.log is False should logs (what would be uploaded to wandb) be saved locally to train/runs/run_name/report_log.txt

config.wandb.project = 'mc_rtt_cont_24' # The wandb project the run should be stored in
config.wandb.sweep_name = 'my-sweep' # The name of the sweep if train.sweep_enabled is True

config.wandb.silent = 'true' # ['true', 'false'] If 'true' wandb does not print anything
config.wandb.alt_wandb_dirs = [ # If the host name is in the list, then store wandb files in the second element of the tuple, * gets replaced with hostname. Useful for reducing load on file system
    (['rock', 'paper', 'scissors'], '/s/*/b/tmp/dmifsud/'),
    (['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'], '/s/*/a/tmp/dmifsud/')]

'''
                    ►──────────── WANDB.SWEEP ────────────◄
'''
# The hyperparameters to search through if config.train.sweep_enabled is True or the --sweep CLI arg is used
# config.wandb.sweep = CN()
# config.wandb.sweep.train = CN()
# config.wandb.sweep.train.warmup_steps = [100, 500, 1000, 2000]
# config.wandb.sweep.train.ramp_start = [50, 100, 500, 1000]
# config.wandb.sweep.train.ramp_end = [500, 1000, 2000]
# config.wandb.sweep.train.epochs = [500, 1000, 2000, 5000]
# config.wandb.sweep.train.init_lr = [0.01, 0.001, 0.005, 0.05]
# config.wandb.sweep.train.batch_size = [64, 512, 1024, 2048, 4096]
# config.wandb.sweep.model = CN()
# config.wandb.sweep.model.hidden_size = [64, 128, 256]
# config.wandb.sweep.model.context_forward = [7, 30]
# config.wandb.sweep.model.context_backward = [7, 30]

config.wandb.sweep = CN()
config.wandb.sweep.setup = CN()
# config.wandb.sweep.setup.seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
config.wandb.sweep.train = CN()
# config.wandb.sweep.train.epochs = [5000, 10000, 20000]
config.wandb.sweep.train.normal_init = [True, False]
# config.wandb.sweep.train.batch_size = [64, 128, 256]
config.wandb.sweep.train.warmup_steps = [1, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 50000, 100000, 1000000]
config.wandb.sweep.train.init_lr = [0.01, 0.001, 0.005, 0.0005, 0.0005, 0.00005, 0.000005]
config.wandb.sweep.train.weight_decay = [0.01, 0.001, 0.0001, 0.00001, 0.00005, 0.000001, 0.000005,  0.0000001, 0.00000001]
# config.wandb.sweep.train.mask_max_span = [1, 2, 3, 4, 5, 6, 7]
# config.wandb.sweep.train.ramp_start = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
# config.wandb.sweep.train.ramp_end = [8000, 9000, 10000, 11000, 15000, 17000, 18000, 20000]
config.wandb.sweep.train.ramp_start = [100, 500, 1000, 2000, 3000]
config.wandb.sweep.train.ramp_end = [3500, 4000, 5000, 7500, 10000, 15000]
config.wandb.sweep.train.mask_max_span = [2, 3, 5, 7]
# config.wandb.sweep.train.add_one_random = [True, False]

config.wandb.sweep.model = CN()
config.wandb.sweep.model.loss_ratio = [0.2, 0.25, 0.35, 0.45, 0.55]
config.wandb.sweep.model.mask_ratio = [0.65, 0.75, 0.85, 0.9]
# config.wandb.sweep.model.random_ratio = [0.85, 1.0]
config.wandb.sweep.model.undivided_attn = [True, False]
# config.wandb.sweep.model.norm = ['layer', 'scale']
config.wandb.sweep.model.initrange = [0.1, 0.01, 0.005, 0.001, 0.0001]
# config.wandb.sweep.model.decoder_layer_size = [32, 64, 128, 256, 512, 1024]
config.wandb.sweep.model.context_forward = [3, 5, 7, 12, 25, 30]
config.wandb.sweep.model.context_backward = [3, 5, 7, 12, 25, 30]
# config.wandb.sweep.model.activation = ['relu', 'gelu']
config.wandb.sweep.model.n_heads = [1, 2, 4]
config.wandb.sweep.model.n_layers = [1, 2, 3, 4, 5]
# config.wandb.sweep.model.emb_size = [0, 32, 64, 128]
config.wandb.sweep.model.dropout_attention = [0.3, 0.4, 0.5, 0.6, 0.7]
config.wandb.sweep.model.dropout_embedding = [0.3, 0.4, 0.5, 0.6, 0.7]
config.wandb.sweep.model.dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
config.wandb.sweep.model.dropout = [0.3, 0.4, 0.5, 0.6, 0.7]
config.wandb.sweep.model.hidden_size = [32, 64, 128, 256, 512]
config.wandb.sweep.model.xavier = [True, False]
'''
────────────────────────────────────────────────────────────────────────────────
                                   FUNCTIONS
'''
def get_config(arg_dict):
    ''' Gets the default config and optionally overwites with values from the
    dataset_config as well as the supplied CLI arguments.

    Args:
        arg_dict (dict): = The command line arguments. Example:
                           {'--name': 'test-run'}
    Returns:
        def_config (CfgNode): The default configuration node, can be treated
                              like a dict.
    '''
    def_config = config.clone()
    if '--default' not in arg_dict: # if '--default' arg is used then config does not get merged with dataset_config
        dataset = arg_dict['--dataset'] if ( # if a dataset is given as an arg, merge with it instead
            '--dataset' in arg_dict
        ) else def_config.setup.dataset
        filename = config.setup.config_dir+'dataset_configs/'+dataset+'.yaml'
        def_config.merge_from_file(filename) # this function overwites values in def_config with those from filename (dataset_config)
    if '--sweep' in arg_dict: def_config.train.sweep_enabled = True # '--sweep' is a shortcut for '--sweep_enabled True'
    for sec in def_config.keys(): # section names
        for key in def_config[sec].keys(): # config parameters
            if '--'+key in arg_dict and key != 'sweep':
                def_config[sec][key] = arg_dict['--'+key] # overwite from arg_dict
    def_config.freeze()
    return def_config

def get_config_from_file(path):
    ''' Gets the default config and merges it with the file from path, default
    config get overwritten.

    Args:
        path (string, Optional): The path to the .yaml file to merge with the
                                 default config.
    Returns:
        file_config (CfgNode): The merged configs.
    '''
    file_config = config.clone()
    file_config.merge_from_file(path)
    file_config.freeze()
    return file_config

def cfg_node_to_dict(cfg_node, key_list=[]):
    ''' Converts the CfgNode to dictionary recursively.

    Args:
        cfg_node (CfgNode): Config.
        key_list (list<string>): Keys for dict.
    Returns:
        cfg_dict (dict): Config dictionary.
    '''
    if not isinstance(cfg_node, CN):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v, key_list + [k])
        return cfg_dict

def get_config_dict(cfg=None):
    ''' Converts config (CfgNode) to dict and returns, for use with wandb.

    Returns:
        config_dict (dict): Config dict.
    '''
    config_copy = config.clone()
    config_copy.freeze()
    config_dict = cfg_node_to_dict(config_copy if not cfg else cfg)
    return config_dict

def get_wandb_config(wandb):
    ''' Overwites the values in the wandb config with those from the wandb sweep
    config. Used when training with a wandb sweep.

    Args:
        wandb (wandb): The wandb import to get the config from.
    Returns:
        config (CfgNode): Config node.
    '''
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            print(k)
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]
    wandb.config = wandb.wandb_sdk.Config()
    for k, v in config.items():
        wandb.config[k] = v
    return config

def get_config_types():
    '''Get the type of all keys in the config, used to convert CLI arguments to
    the type used in the config.

    Returns:
        type_dict (dict): A dict containing the config parameters as keys and
                          the type as the value.
    '''
    type_dict = {}
    def_config = config.clone()
    meta_list = [list(def_config.get(i).items()) for i in ['train', 'model', 'setup', 'wandb']]
    cat_list = meta_list[0]+meta_list[1]+meta_list[2]+meta_list[3]
    for key, value in cat_list:
        type_dict[key] = type(value)
    return type_dict
















        #
