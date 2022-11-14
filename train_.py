#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
# from test_eval import test_eval
from timeit import default_timer as timer
import os
# import gc
import sys
import time
import shutil

from glob import glob
# #────#
import torch
import wandb
import numpy as np
import torch.nn as nn
from nlb_tools.evaluation import bits_per_spike as bits_per_spike2
import subprocess

# #────#
from transformer import Transformer
# from transformer_deberta import Transformer
from utils_f import (get_config,
                   metric_comparison,
                   bits_per_spike,
                   get_config_dict,
                   get_config_from_file,
                   get_run_name,
                   set_seeds,
                   parse_args,
                   set_device,
                   get_wandb_config,
                   get_optimizer,
                   add_tmux_agents,
                   get_scheduler,
                   get_wandb_dir,
                   set_sweep_config,
                   set_sweep_config2,
                   setup_runs_folder,
                   print_train_configs,
                   start_tmux_sweep,
                   upload_print_results,
                   print_run_get_prog_bar,
                   wandb_cleanup)
# from configs.default_config import (get_config,
# #                                     get_config_dict,
# #                                     get_wandb_config,
#                                     get_config_from_file)
'''──────────────────────────────── train.py ────────────────────────────────'''
# This file train can train a single NDT-U model, start a hyper parameter sweep,
# or add an agent to a hyperparameter sweep. It has optional arguments that can
# be found by running 'python train.py -h'.

# from datasets import verify_dataset
from datasets import get_dataloaders, get_alignment_matricies
from yacs.config import CfgNode as CN

# Will beome 'train.py'
import signal
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from test import test_func
import yaml

from utils.data.create_local_t5data import get_trial_data


def main():

    # Parse arguments
    arg_dict = parse_args(sys.argv[1:])
    # Overwrite default config with CLI args and dataset_config
    config = get_config(arg_dict)

    # If data does not exist, download it
    # verify_dataset(config) # makes sure datasets are downloaded, if not then prompt
    print_train_configs(config, arg_dict) # prints the config if starting a single run or sweep
    model_name = arg_dict['--name'] # if no name arg was passed then this is None

    # Set GPU used by cuda:0 from config.setup.gpu_idx
    set_device(config, arg_dict)
    device = torch.device('cuda:0')

    os.environ['WANDB_SILENT'] = 'true' if config['wandb']['silent'] else 'false' # dont print wandb logs

    # Add an agent to a wandb sweep
    if '--add' in arg_dict:
        add_sweep_agent(config)
    elif '--tadd' in arg_dict:
        add_tmux_agents(config['setup']['ag_gpus'])

    # Start a wandb sweep
    elif config['train']['sweep_enabled'] or '--sweep' in arg_dict:
        sweep_id = set_sweep_config2(config, arg_dict)
        add_sweep_agent(config, sweep_id)
        # Remove wandb sweep folder when completed
        shutil.rmtree(glob('./wandb/*'+sweep_id+'/')[0])
    elif '--tsweep' in arg_dict:
        # Start tmux session then start sweep and add agents
        start_tmux_sweep(config['setup']['ag_gpus'])

    # Run training
    else:
        run_training(config, device, model_name)


def run_training(config, device, name):
    set_seeds(config)
    cross_val_enabled = (config['train']['val_type'] == 'cross_val')

    if config['wandb']['log']:
        # initialize wandb run
        wandb.init(
            dir=get_wandb_dir(),
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config={})

        # if in a sweep, update the config
        if wandb.run.sweep_id != None:
            for str_name in wandb.config.keys():
                group, key = str_name.split('.')
                config[group][key] = wandb.config[str_name]

        # upload final config to wandb
        wandb.config.update(get_config_dict(config), allow_val_change=True)
        
    #     config = CN(get_wandb_config())

    # name = get_run_name(config, name)

    # train_dataloader, val_dataloader = get_dataloaders(
    #     config, 
    #     'train_val'
    #     # 'cross_val' if cross_val_enabled else 'train_val'
    # )

    # if cross_val_enabled:
    #     for idx, train_sub_dl, val_sub_dl in val_dataloader:
    #         run_name = f'{name}_f{idx}'
    #         dataset_sub = train_sub_dl.dataset.dataset
    #         model = Transformer(config, dataset_sub, run_name, device).to(device)
    #         train(model, train_sub_dl, val_sub_dl, device, f'_f{idx}')
    #     val_dataloader = None
    #     dataset = train_dataloader.dataset 
    # else:
    #     dataset = train_dataloader.dataset.dataset 

    # model = Transformer(config, dataset, name, device).to(device)
    # train(model, train_dataloader, val_dataloader, device)
    # test_func(config, model)

def add_sweep_agent(config, id=None):
    '''Adds an agent to a wandb sweep. If no id is supplied then prompt the user
    to choose from all current sweeps.

    Args:
        config (dict): The config to be used.
        id (str, Optional): The id of the sweep to add this agent to.
    '''
    if id == None: # prompt the user from all current sweeps
        file_list = glob('./wandb/sweep*')
        id = ''
        id_list = []
        options = 1
        if len(file_list) > 1:
            for file in file_list:
                last_edit = time.strftime("%m/%d/%y %I:%M%p", time.localtime(os.path.getctime(file)))
                sweep_id = file.split('/')[-1].split('-')[-1]
                id_list.append(sweep_id)
                print(str(options)+': '+sweep_id+' - '+last_edit)
                options+=1
            chosen_index = input('\nWhich sweep ID should the sweep agent use? (1-'+str(len(id_list))+'): ')
            while int(chosen_index) not in range(1,len(id_list)+1):
                chosen_index = input('\nPlease enter a number between 1 and '+str(len(id_list))+': ')
            id = id_list[int(chosen_index)-1]
        else:
            try: id = file_list[0].split('/')[-1].split('-')[-1]
            except: 
                print('No sweeps are currently running.')
                exit()

    print('Adding agent to sweep with ID:', id+'\n')
    my_env = os.environ.copy()
    if config['setup']['gpu'] != -1:
        my_env["CUDA_VISIBLE_DEVICES"] = str(config['setup']['gpu'])

    call = [
        "wandb", "agent", 
        "-p", f"{config['wandb']['project']}", 
        "-e", f"{config['wandb']['entity']}", 
        f'{id}'
    ]
    if config['train']['sweep_type'] != 'grid':
        call.insert(2, '--count')
        call.insert(3, f'{config["train"]["sweep_epochs"]}')

    with subprocess.Popen(call, env=my_env) as p:
        try: return p.wait()
        except:  
            p.send_signal(signal.SIGINT)
            p.wait()


if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        main()
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')