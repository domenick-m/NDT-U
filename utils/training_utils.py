#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import os.path as osp
import shutil
import random
import tqdm
import argparse
import yaml
import subprocess
from glob import glob
from distutils.util import strtobool
#────#
import torch
import torch.nn as nn
import numpy as np
import wandb
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
#────#
from utils.config_utils import get_config_types, cfg_node_to_dict

'''─────────────────────────── training_utils.py ____________________________'''
# This file contains ...


def nll(rates, spikes):
    if torch.any(torch.isnan(spikes)):
        mask = torch.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]
    
    assert not torch.any(torch.isnan(rates)), \
        "neg_log_likelihood: NaN rate predictions found"

    assert torch.all(rates >= 0), \
        "neg_log_likelihood: Negative rate predictions found"
    if (torch.any(rates == 0)):
        rates[rates == 0] = 1e-9
    return nn.functional.poisson_nll_loss(rates, spikes, log_input=False, full=True, reduction='sum')


def bits_per_spike(rates, spikes):
    ch_means = torch.nanmean(spikes, dim=(0,1), keepdim=True).tile((spikes.shape[0], spikes.shape[1], 1))
    
    nll_model = nll(rates, spikes)
    nll_null = nll(ch_means, spikes)
    
    lt_nll_model = nll(rates[:, -1, :], spikes[:, -1, :])
    lt_nll_null = nll(ch_means[:, -1, :], spikes[:, -1, :])

    bps = float((nll_null - nll_model) / spikes.nansum() / 0.6931471805599453)
    lt_bps = float((lt_nll_null - lt_nll_model) / spikes[:, -1, :].nansum() / 0.6931471805599453)
    return bps, lt_bps


def parse_args():
    '''Parses the CLI args.

    Args:
        args (list<str>): sys.argv{1:}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', action='store_true',
                        help='This is a test that is a test that is a test that is a test.')
    parser.add_argument('--silent', action='store_true',
                        help='This is a test that is a test that is a test that is a test.')
    parser.add_argument('--sweep', action='store_true',
                        help='This is a test that is a test that is a test that is a test.')
    parser.add_argument('--add', action='store_true',
                        help='This is a test that is a test that is a test that is a test.')
    parser.add_argument('--tmux', action='store_true',
                        help='This is a test that is a test that is a test that is a test.')
    parser.add_argument('--name', type=str, default=None,
                        help='This is a test that is a test that is a test that is a test.')
    args, cfg_args = parser.parse_known_args()
    args = vars(args)

    type_dict = get_config_types()
    for index, arg in enumerate(cfg_args):
        if arg[:2] == '--':
            param = arg[2:]
            # assert parameter is in config
            assert param in type_dict, \
                f'\n\n! Invalid Argument. !\n  {arg} is not recognized.\n  {"‾"*len(arg)}\nuse -h to show help message.'
            # assert that config has value
            assert len(cfg_args) > index+1, \
                f'\n\n! Argument Missing Value. !\n  {arg} is missing a value.\n  {"‾"*len(arg)}'
            try:
                if type_dict[param] == bool: 
                    args[param] = type_dict[param](strtobool(cfg_args[index+1]))

                elif type_dict[param] == list: 
                    args[param] = [int(i) for i in cfg_args[index+1].strip('][').split(',')]
                    
                else: args[param] = type_dict[param](cfg_args[index+1])
            except:
                true_type = f'{str(type_dict[param])}\n{" "*(len(arg)+3)}{"‾"*len(cfg_args[index+1])}'
                raise Exception(
                    f'\n! Argument Type Error. !\n  {arg} {cfg_args[index+1]} ← needs to be type: {true_type}')
    return args


def set_seeds(config):
    ''' Sets seeds.

    Args:
        config (dict): A config object.
    '''
    seed = config.train.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_device(config, arg_dict):
    ''' Sets torch device according to config.setup.gpu_idx, -1 will auto-select
    the GPU with the least memory usage.

    Args:
        config (dict): A config object.
    '''    
    if 'gpu' in arg_dict:
        device = arg_dict['gpu'] # CLI defined GPU index
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        return
    else:
        # If gpu_idx = -1 then auto-select the least used GPU
        if config.train.gpu == -1:
            smi_output = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
            gpu_list = smi_output.decode('utf-8').split('\n')
            gpu_list = list(filter(lambda x: 'Used' in x, gpu_list))
            gpu_list = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_list] # list of memory usage in MB
            device = str(min(range(len(gpu_list)), key=lambda x: gpu_list[x])) # get argmin
        else: device = str(config.train.gpu) # user defined GPU index
    # Using env vars allows cuda:0 to be used regardless of GPU
    if 'tmux' in arg_dict and not arg_dict['tmux']:
        os.environ['CUDA_VISIBLE_DEVICES'] = device


def create_model_dir(config, name):
    if name is None: # this means nothing was passed to name CLI arg
        name = wandb.run.name if config.log.to_wandb else input('\nEnter an ID to save the model with: ')
    
    os.makedirs(config.dirs.save_dir, exist_ok=True)
    path = osp.join(config.dirs.save_dir, name)

    n_runs = len(glob(path)) + len(glob(f'{path}_*'))
    if n_runs > 0:
        name += f'_{n_runs}'
        path = osp.join(config.dirs.save_dir, name)
        print(f'\n! RENAMING !\nName is now: {name}\n')

    os.makedirs(path, exist_ok=False) # create run dir

    with open(osp.join(path, 'config.yaml'), 'w') as yfile:
        yaml.dump(cfg_node_to_dict(config), yfile)
        
    if config.log.to_wandb:
        wandb.run.name = name
    
    return path


def get_swept_params(sweep_cfg_dir):
    with open(sweep_cfg_dir, 'rb') as yamlf:
        yaml_dict = yaml.load(yamlf)
        param_list = [i.split('.')[1] for i in yaml_dict['parameters'].keys()]
        value_list = [str(i) for i in yaml_dict['parameters'].values()]
    return param_list, value_list


def create_progress_bar(config, model, wandb=None):
    f_width = shutil.get_terminal_size((80,0)).columns
    width = int(f_width / 3)
    w_1 = int(width*0.8)
    w_2 = w_1 + f_width - (w_1 * 3 + 4)

    if config.log.to_wandb:
        print('wandb URL:', wandb.run.get_url(), '\n')

    params = f'n_parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad):,}'
    gpu_idx = f'GPU:{os.environ["CUDA_VISIBLE_DEVICES"]}'.center(w_1)

    name = osp.basename(config.dirs.save_dir)
    print(f'\n{name:{w_1}}--{gpu_idx}--{params:>{w_2}}\n', file=sys.stderr)
    bar_format = ('Epochs: {n_fmt} / %s {bar} {percentage:3.0f}%% - ETR:{remaining}' % config.train.epochs)

    return tqdm(
        range(config.train.epochs),
        unit='epochs',
        desc='Epochs',
        position=2,
        bar_format=bar_format
    )


def print_train_configs(config, args):
    ''' Prints the configs on train startup.

    Args:
        config (dict): A config object.
        args (dict): The CLI args in dict form.
    '''
    if args['add'] or args['silent']:
        return None

    width = shutil.get_terminal_size((80,0)).columns
    w_1 = int(width / 3)
    w_2 = w_1 + (width - (w_1 * 3 + 2))

    if width < 94:
        print(f'\n ! TERMINAL TOO SMALL TO PRINT CONFIGS !\nWidth needs to be at least 94... Current Size: {width}')
        return None

    def title_box(sec, width):
        sec_len = len(sec)
        pad = ' ' if sec_len == 3 else ''
        mid_pad = '' if sec_len % 2 == 1 else ' '
        return (
            f'╔{"═"*11}╗'.center(width), 
            f'╣   {pad}{sec}{mid_pad}{pad}   ╠'.center(width, '═'), 
            f'╚{"═"*11}╝'.center(width)
        )

    if args['sweep']:
        swept_params, swept_values = get_swept_params(config.dirs.sweep_cfg_path)

    def format_config(sec, param, max_val_width, max_width):
        if args['sweep'] and param in swept_params:
            return f'{param+":":<{max_width+1}}   {"(sweep)".ljust(max_val_width)}'
            # return f'{"     "+param+":":<{max_width+1}}   {"(sweep)".ljust(max_val_width)}'
        else: 
            return f'{param+":":<{max_width+1}}   {str(config[sec][param]).ljust(max_val_width)}'
            # return f'{"     "+param+":":<{max_width+1}}   {str(config[sec][param]).ljust(max_val_width)}'

    s_len = lambda x: len(str(x))
    max_sec_lens = {sec: max([s_len(i) for i in config[sec].keys() if i != "sessions"]) for sec in config.keys()}
    max_val_lens = {sec: max([s_len(j) for i,j in config[sec].items() if i != "sessions"]) for sec in config.keys()}
    config_dict = {sec: [i for i in title_box(sec,  w_2 if sec == 'train' else w_1)] for sec in config.keys() if sec != 'dirs'}

    w_1_30 = int(w_1 * 0.3)
    w_2_30 = int(w_2 * 0.3)

    max_l = lambda x: max(w_2_30 if x == 'train' else w_1_30, max_sec_lens[x]+1)
    max_p = lambda x: int(width * 0.3) if x + int(width * 0.3) < width else 0

    for sec in config.keys():
        if sec != 'dirs':
            for param in config[sec].keys():
                if param != 'sessions' and config[sec][param] != '':
                    config_dict[sec].append(format_config(sec, param, max_val_lens[sec], max_l(sec)).center(w_2 if sec == 'train' else w_1))
            if sec == 'data':
                config_dict[sec].append('')

    title = 'Pre-training' if config.dirs.trained_mdl_path == '' else ' Fine-tuning'
    
    ts_1 = ts_2 = int((width - 26) / 2)
    if width % 2 == 0: ts_1 -= 1

    #
    # ! START PRINT !
    #

    print(f'\n{" "*ts_1}╔═════════════════════════╗')
    print(f'{"═"*ts_1}╣  {title} Model...  ╠{"═"*ts_2}')
    print(f'{" "*ts_1}╚═════════════════════════╝')

    print(f'\nSessions:')
    for i in config.data.sessions:
        print('  '+i)
    print('')

    for i in title_box('dirs',  width):
        print(i)
    for i in config.dirs.keys():
        print(format_config('dirs', i, max_val_lens['dirs']+max_p(max_val_lens['dirs']+max_l('dirs')), max_l('dirs')).center(width))
    print('')

    row = [config_dict['data']+config_dict['log'], config_dict['train'], config_dict['model']]

    len_list = [len(x) for x in row]
    for length, list in zip(len_list, row):
        for i in range(max(len_list) - length):
            list.append('')

    for idx, ijk in enumerate(zip(row[0], row[1], row[2])):
        i, j, k = ijk
        symb1 = "╦" if "data" in i else "╣" if "log" in i else "║" if idx != 0 else " "
        symb2 = "╦" if "train" in j else "║" if idx != 0 else " "
        print(f'{i:{w_1}}{symb1}{j:{w_2}}{symb2}{k:{w_1}}')
        # print(f'{i:{w_1+1 if i == "" and width % 2 == 0 else w_1}}{symb1}{j:{w_2}}{symb2}{k:{w_1}}')

    if args['sweep']:
        print('Sweep Parameters:')
        max_length = max(len(x) for x in swept_params)
        for name, value in zip(swept_params, swept_values):
            print(f'  {name:>{max_length}}: {value}')

    # print(f'{" "*w_1}║{" "*w_2}║{" "*w_1}')
    print(f'{"═"*w_1}╩{"═"*w_2}╩{"═"*w_1}')

    # Confirm
    if not args['y']:
        response = input('Proceed? (y/n): ')
        while response not in ['y', 'n']:
            response = input("Please enter 'y' or 'n': ")
        if response == 'n': exit()