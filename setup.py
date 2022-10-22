#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import math
import random
import subprocess
from glob import glob
from distutils.util import strtobool
#────#
import yaml
import wandb
import torch
import socket
import shutil
import numpy as np
import torch.nn as nn
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from torch.optim.lr_scheduler import LambdaLR
#────#
from configs.default_config import (get_config_types,
                                    cfg_node_to_dict,
                                    get_config)
'''──────────────────────────────── setup.py ────────────────────────────────'''
# This is for misc functions that set up stuff.

def set_device(config):
    ''' Sets torch device according to config.setup.gpu_idx, -1 will auto-select
    the GPU with the least memory usage.

    Args:
        config (dict): A config object.
    '''
    device = None
    # If gpu_idx = -1 then auto-select the least used GPU
    if config['setup']['gpu_idx'] == -1:
        smi_output = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
        gpu_list = smi_output.decode('utf-8').split('\n')
        gpu_list = list(filter(lambda x: 'Used' in x, gpu_list))
        gpu_list = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_list] # list of memory usage in MB
        device = str(min(range(len(gpu_list)), key=lambda x: gpu_list[x])) # get argmin
    else: device = str(config['setup']['gpu_idx']) # user defined GPU index
    # Using env vars allows cuda:0 to be used regardless of GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = device

def set_seeds(config):
    ''' Sets seeds.

    Args:
        config (dict): A config object.
    '''
    seed = config['setup']['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_sweep_config(config, arg_dict):
    '''Starts a sweep with wandb.sweep and creates a folder with the sweeps
    config in it. Returns the id of the created sweep.

    Args:
        config (dict): A config object.
        arg_dict (CfgNode): Dictionary of the CLI args used when calling train.
    Returns:
        sweep_id (int): The sweep id.
    '''
    sweep_dict = {}
    wandb_sweep = config['wandb']['sweep']
    empty = True
    for group in wandb_sweep.keys():
        for key in wandb_sweep[group].keys():
            if key: empty = False
            values = wandb_sweep[group][key].copy()
            sweep_dict[group+'.'+key] = {'values':values}
    if empty:
        print("Cannot start a sweep, please add values under 'config.wandb.sweep'.")
        exit()
    sweep_config = {
        'name' : config['wandb']['sweep_name'],
        'method' : config['train']['sweep_type'],
        'parameters' : sweep_dict
    }
    sweep_id = wandb.sweep(sweep_config, project=config['wandb']['project'])
    path = './wandb/sweep-'+sweep_id+'/'
    os.makedirs(path)
    config['wandb']['alt_wandb_dirs'] = []
    with open(path+'/config.yaml', 'w') as yamlfile:
        data = yaml.dump(cfg_node_to_dict(config), yamlfile)
    return sweep_id

def setup_runs_folder(config, model, mode):
    '''Creates a folder for a run.

    Args:
        config (dict): A config object.
        model (Transformer): The model.
        mode (str): ['train', 'test'] The run type.
    Returns:
        path (str): The path to the created folder.
    '''
    path = config['setup']['runs_dir'] + mode + '/' + model.name
    if os.path.isdir(path):
        file_count = 1
        tmp_path = path
        while os.path.isdir(tmp_path):
            tmp_path = path+'_'+str(file_count)
            file_count += 1
        path = tmp_path
        new_name = model.name+'_'+str(file_count-1)
        print('\n RENAMING TO:', new_name+'\n')
        model.name = new_name
    os.makedirs(path)
    config['wandb']['alt_wandb_dirs'] = []
    with open(path+'/config.yaml', 'w') as yamlfile:
        data = yaml.dump(cfg_node_to_dict(config), yamlfile)
    return path+'/'

def delete_wandb_run_folder(wandb):
    '''Deletes the run folder after it's been uploaded.

    Args:
        wandb (wandb): The imported wandb.
    '''
    if wandb.run != None:
        run_id = wandb.run.id
        wandb.finish()
        dir_q = get_wandb_dir()
        wandb_dir = dir_q if dir_q != None else '.'
        shutil.rmtree(glob(wandb_dir+'/wandb/*'+run_id+'/')[0])

def get_wandb_dir():
    '''Gets the wandb directory, could be different for different hosts.
    '''
    wandb_dir = None
    config = get_config({})
    hostname = socket.gethostname()
    alt_wandb_dirs = config['wandb']['alt_wandb_dirs']
    if len(alt_wandb_dirs) > 0:
        for host_list, dir in alt_wandb_dirs:
            if hostname in host_list:
                dir_list = dir.split('*')
                wandb_dir = dir_list[0] + hostname + dir_list[1]
    return wandb_dir

def get_optimizer(model, config):
    ''' Gets the optimizer.

    Args:
        model (Transformer): The model.
        config (dict): A config object.
    '''
    parameters = model.parameters()
    if config['train']['optimizer'] == 'AdamW':
        return torch.optim.AdamW(parameters,
            lr=config['train']['init_lr'], # Default is 0.001
            weight_decay=config['train']['weight_decay']) # Default is 0.01
    # elif config['train']['optimizer'] == 'new optimizer':
    #     return torch.optim.AdamW(parameters,)

def get_norm(config, n_neurons):
    ''' Gets the normalization.

    Args:
        n_neurons (int): Number of neurons.
        config (dict): A config object.
    '''
    if config['model']['norm'] == 'layer':
        return nn.LayerNorm(n_neurons)
    elif config['model']['norm'] == 'scale':
        return ScaleNorm(n_neurons**0.5)
    elif config['model']['norm'] == 'switch':
        return SwitchNorm2d(n_neurons)

class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        x = x.permute(1, 2, 0) # t b n -> b n t
        N, C, T = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, T)
        return (x * self.weight + self.bias).permute(2, 0, 1) # b n t -> t b n

class ScaleNorm(nn.Module):
    '''ScaleNorm'''
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


def get_scheduler(optimizer, config):
    ''' Gets the scheduler.

    Args:
        optimizer (optimizer): The optimizer to schedule.
        config (dict): A config object.
    '''
    scheduler = None
    if config['train']['scheduler'] == 'Cosine':
        scheduler = WarmupCosineSchedule(optimizer,
            warmup_steps=config['train']['warmup_steps'],
            t_total=config['train']['epochs'])
    # elif config['train']['scheduler'] == 'new scheduler':
    #     scheduler = new scheduler(optimizer,)
    return scheduler

class WarmupCosineSchedule(LambdaLR):
    '''WarmupCosineSchedule'''
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def parse_args(args):
    '''Parses the CLI args.

    Args:
        args (list<str>): sys.argv{1:}
    '''
    arg_dict = {'--name': None}
    type_dict = get_config_types()
    for index, arg in enumerate(args):
        if arg == '-h' or  arg == '--help':
            print('usage: train.py [-h] [-y] [--add] [--sweep] [--default] [--name name] [--{CFG KEY} value]\n\nTrains a model.\n')
            print('options:')
            print(' -h, --help         show this help message and exit')
            print(' -y                 skip the confirmation dialog')
            print(' --add              adds an agent to an already existing sweep')
            print(' --sweep            starts a sweep, identical to --sweep_enabled True')
            print(' --default          only the default_config should be used (no dataset_configs)')
            print(' --name name        assigns given name to the run, used in wandb and when saving')
            print(' --{CFG KEY} value  can change any setting in the config to the supplied value')
            print('\nexample:\n train.py -y --dataset area2_bump')
            exit()# NOTE:
        elif arg in ['-y', '--add', '--sweep', '--default']:
            arg_dict[arg] = True
        elif arg == '--name':
            if args[index+1][:2] != '--':
                arg_dict[arg] = args[index+1]
            else:
                print('\n! Argument Missing Value. !\n  '+arg+' is missing a value.\n  '+'‾'*len(arg))
                exit()
        elif arg[:2] == '--':
            if arg[2:] not in type_dict:
                print('\n! Invalid Argument. !\n  '+arg+' is not recognized.\n  '+'‾'*len(arg))
                print('use -h to show help message.')
                exit()
            if len(args) <= index+1:
                print('\n! Argument Missing Value. !\n  '+arg+' is missing a value.\n  '+'‾'*len(arg))
                exit()
            try:
                arg_dict[arg] = type_dict[arg[2:]](strtobool(args[index+1])) if (
                    type_dict[arg[2:]] == bool
                ) else type_dict[arg[2:]](args[index+1])
            except:
                print('\n! Argument Type Error. !')
                print('  '+arg+' '+args[index+1]+' ← needs to be type: '+str(type_dict[arg[2:]]))
                print(' '*(len(arg)+3) + '‾'*len(args[index+1]))
                exit()
    return arg_dict

def print_run_get_prog_bar(config, model, wandb=None):
    '''Prints run info and creates the tqdm bar.

    Args:
        config (dict): A config object.
        model (Transformer): The model.
        wandb (wandb, Optional): wandb after import.
    Returns:
        tqdm (tqdm): The progress bar.
    '''
    params = str('{:,}'.format(sum(param.numel() for param in (
        model.parameters()) if param.requires_grad)))
    f_width = shutil.get_terminal_size((80,0)).columns
    width = int(f_width / 3)
    w_1 = int(width*0.8)
    w_2 = w_1 + f_width - (w_1 * 3 + 4)
    if config['wandb']['log']:
        print('wandb URL:', wandb.run.get_url(), '\n')
    print('\n{0:{1}}--{2}--{3:>{4}}\n'.format(
        model.name, w_1,
        ('GPU:'+os.environ['CUDA_VISIBLE_DEVICES']).center(w_1),
        'n_parameters: '+params, w_2
    ))
    bar_format = ('Epochs: {n_fmt} / %s {bar} {percentage:3.0f}%% - ETR:{remaining}' % config['train']['epochs'])
    return tqdm(
        range(config['train']['epochs']),
        unit='epochs',
        desc='Epochs',
        position=2,
        bar_format=bar_format
    )

def upload_print_results(config, result_dict, progress_bar, save_path):
    '''Either uploads to wandb or stores locally. Also shows the report on the
    progress bar.

    Args:
        config (dict): A config object.
        result_dict (dict): Val set results.
        progress_bar (tqdm): Progress bar.
        save_path (str): Where to save locally.
    '''
    epoch = '[Epoch: '+result_dict['epoch']+']'
    loss = '[val loss: '+"{:.4f}".format(result_dict['val_loss'])+']'
    cobps = '[val co-bps: '+"{:.3f}".format(result_dict['co_bps'])+']'
    fpbps = '[val fp-bps: '+"{:.3f}".format(result_dict['fp_bps'])+']'
    report = epoch + '   ' + loss + '   ' + cobps + '   ' + fpbps
    if config['wandb']['log']:
        wandb.log({
            'val loss': result_dict['val_loss'],
            'val heldout loss': result_dict['heldout_loss'],
            'val co-bps': result_dict['co_bps'],
            'val forward loss': result_dict['forward_loss'],
            'val fp-bps': result_dict['fp_bps'],
            'val heldin loss': result_dict['heldin_loss']
        })
    elif config['wandb']['log_local']:
        with open(save_path+'report_log.txt', 'a') as f:
            f.write('\n'+report)
            f.write(']\n'+' '*13+'[val heldin loss: '+ "{:.3f}".format(result_dict['heldin_loss']))
            f.write(']  [val heldout loss: '+ "{:.3f}".format(result_dict['heldout_loss']))
            f.write(']  [val forward loss: '+ "{:.3f}".format(result_dict['forward_loss'])+']')
    progress_bar.display(msg=report, pos=0)

def print_train_configs(config, args):
    ''' Prints the configs on train startup.

    Args:
        config (dict): A config object.
        args (dict): The CLI args in dict form.
    '''
    if '--add' in args:
        return None

    def format_config(name, value):
        ''' Helper used by print_train_configs.
        '''
        if value == '_box_':
            return ('┌'+'─'*11+'┐', '│   '+name+'   │', '└'+'─'*11+'┘')
        if not config.train.sweep_enabled:
            return name+': '+str(value)
        elif name in [config.wandb.sweep[x].keys() for x in [
            'train', 'model', 'setup'
        ]]:
            return name+': (sweep)'
        else: return name+': '+str(value)

    setup_box = format_config('setup', '_box_')
    wandb_box = format_config('wandb', '_box_')
    model_box = format_config('model', '_box_')
    train_box = format_config('train', '_box_')

    setup_list = [
        setup_box[0], setup_box[1], setup_box[2],
        format_config('dataset', config.setup.dataset),
        format_config('seed', config.setup.seed),
        format_config('log_eps', config.setup.log_eps),
        format_config('save_model', config.setup.save_model),
        format_config('save_min_bps', config.setup.save_min_bps),
        '', wandb_box[0], wandb_box[1], wandb_box[2],
        format_config('project', config.wandb.project),
        format_config('log', config.wandb.log),
        format_config('log_local', config.wandb.log_local),
        format_config('log_freq', config.wandb.log_freq)
    ]
    model_list = [
        model_box[0], model_box[1], model_box[2],
        format_config('n_layers', config.model.n_layers),
        format_config('n_heads', config.model.n_heads),
        format_config('hidden_size', config.model.hidden_size),
        format_config('dropout', config.model.dropout),
        format_config('dropout_rates', config.model.dropout_rates),
        format_config('dropout_embedding', config.model.dropout_embedding),
        format_config('dropout_attention', config.model.dropout_attention),
        format_config('activation', config.model.activation),
        format_config('norm', config.model.norm),
        format_config('initrange', config.model.initrange),
        format_config('context_forward', config.model.context_forward),
        format_config('context_backward', config.model.context_backward),
        format_config('loss_ratio', config.model.loss_ratio),
        format_config('mask_ratio', config.model.mask_ratio),
        format_config('random_ratio', config.model.random_ratio)
    ]
    train_list = [
        train_box[0], train_box[1], train_box[2],
        format_config('batch_size', config.train.batch_size),
        format_config('epochs', config.train.epochs),
        format_config('early_stopping', config.train.early_stopping),
        format_config('es_min_bps', config.train.es_min_bps),
        format_config('optimizer', config.train.optimizer),
        format_config('scheduler', config.train.scheduler),
        format_config('warmup_steps', config.train.warmup_steps),
        format_config('init_lr', config.train.init_lr),
        format_config('warmup_steps', config.train.warmup_steps),
        format_config('weight_decay', config.train.weight_decay),
        format_config('mask_max_span', config.train.mask_max_span),
        format_config('ramp_start', config.train.ramp_start),
        format_config('ramp_end', config.train.ramp_end),
        format_config('val_type', config.train.val_type)
    ]

    if config.train.val_type == 'random':
        train_list.append(format_config('val_ratio', config.train.val_ratio))
        setup_list.insert(5, format_config('subset_seed', config.setup.subset_seed))

    if not config.train.sweep_enabled: setup_list.append('sweep: Disabled')
    else: setup_list.extend(['', 'sweep: Enabled'])

    width = shutil.get_terminal_size((80,0)).columns
    w_1 = int((width - 26) / 2)
    print('\n'+' '*w_1+'┌────────────────────────┐')
    print(' '*w_1+'│   Running training...  │')
    print(' '*w_1+'└────────────────────────┘')
    print('configs used:\n    /'+config.setup.config_dir+'default.py')
    print('    /'+config.setup.config_dir+'dataset_configs/'+config.setup.dataset+'.yaml')

    w_1 = int((width - 9) / 2)
    w_2 = w_1 + (width - (w_1 * 2 + 9))
    print('\n'+'─'*w_1+' configs '+'─'*w_2+'\n')

    row = [setup_list, model_list, train_list]

    len_list = [len(x) for x in row]
    for length, list in zip(len_list, row):
        for i in range(max(len_list) - length):
            list.append('')

    w_1 = int(width / 3)
    w_2 = w_1 + (width - (w_1 * 3 + 2))
    for i, j, k in zip(row[0], row[1], row[2]):
        print('{0:{1}} {2:{3}} {4:{5}}'.format(i, w_1, j, w_2, k, w_1))
    if config.train.sweep_enabled:
        param_names, param_values = [], []
        print('Sweep Type:', config.train.sweep_type)
        print('Sweep Parameters:')
        for param_group in config.wandb.sweep:
            for param in config.wandb.sweep[param_group]:
                param_names.append(param)
                param_values.append(str(config.wandb.sweep[param_group][param]))
        if len(param_names) == 0: print('  ! EMPTY !')
        else:
            max_length = max(len(x) for x in param_names)
            for name, value in zip(param_names, param_values):
                print('  {0:>{1}}: {2}'.format(name, max_length, value))
    print('\n'+('─'*width)+'\n')

    # Confirm
    if '-y' not in args:
        response = input('Proceed? (y/n): ')
        while response not in ['y', 'n']:
            response = input("Please enter 'y' or 'n': ")
        if response == 'n': exit()
