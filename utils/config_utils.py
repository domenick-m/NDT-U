#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os.path as osp
import sys
sys.path.append('../')
#────#
from yacs.config import CfgNode as CN
#────#
from def_config_ import config
import wandb
'''──────────────────────────── config_utils.py _____________________________'''
# This file contains ...


def get_config(arg_dict=None):
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
    
    # Overwites values from ...
    if config.dirs.cfg_path != '' and osp.isfile(config.dirs.cfg_path):
        def_config.merge_from_file(config.dirs.cfg_path) 

    if arg_dict != None:
        for sec in def_config.keys(): # section names
            for key in def_config[sec].keys(): # config parameters
                if key in arg_dict:
                    def_config[sec][key] = arg_dict[key] # overwite from arg_dict
    
    def_config.freeze()
    return def_config


def get_config_types():
    '''Get the type of all keys in the config, used to convert CLI arguments to
    the type used in the config.

    Returns:
        type_dict (dict): A dict containing the config parameters as keys and
                          the type as the value.
    '''
    def_config = config.clone()

    meta_list = []
    for i in def_config.keys():
        meta_list += list(def_config.get(i).items()) 

    type_dict = {}
    for key, value in meta_list:
        type_dict[key] = type(value)
    return type_dict


def verify_config(config, arg_dict):
    '''
    Args:
    Returns:
    '''
    assert not (config.dirs.trained_mdl_path != '' and not osp.isfile(config.dirs.trained_mdl_path)), (
        f'\n\nError: ! Pre-trained model state dictionary not found!. !\nThe file {config.dirs.trained_mdl_path} does'
        ' not exist')

    assert not (arg_dict['sweep'] and config.dirs.sweep_cfg_path == ''), (
        '\n\nError: ! Missing path to sweep config. !\ntrain.py was called with "--sweep" however no sweep config path '
        'was specified in config.dirs.sweep_cfg_path')


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

def update_config_from_sweep(config):
    ''' 
    '''
    # overwrite the original config used in the sweep 
    config = get_config_from_file(f'./wandb/sweep-{wandb.run.sweep_id}/config.yaml')
    # the wandb config will only contain the values being swept over
    for str_name in wandb.config.keys():
        # wandb cannot use nested parameters for sweeps, dots are used to seperate
        group, key = str_name.split('.')
        config[group][key] = wandb.config[str_name] 

    return config

