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
from transformer_ import Transformer
# from transformer_deberta import Transformer
# from utils_f import (get_config,
#                    metric_comparison,
#                    bits_per_spike,
#                    get_config_dict,
#                    get_config_from_file,
#                    get_run_name,
#                    set_seeds,
#                    parse_args,
#                    set_device,
#                    get_wandb_config,
#                    get_optimizer,
#                    add_tmux_agents,
#                    get_scheduler,
#                    get_wandb_dir,
#                    set_sweep_config,
#                    set_sweep_config2,
#                    setup_runs_folder,
#                    print_train_configs,
#                    start_tmux_sweep,
#                    upload_print_results,
#                    print_run_get_prog_bar,
#                    wandb_cleanup)
# from configs.default_config import (get_config,
# #                                     get_config_dict,
# #                                     get_wandb_config,
#                                     get_config_from_file)


# from datasets import verify_dataset
from yacs.config import CfgNode as CN

import os.path as osp
# Will beome 'train.py'
import sys
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from test import test_func
import yaml

# from create_local_t5data import (
#     get_trial_data)
from utils.t5_utils import (
    get_pretraining_data)
from utils.data_utils import (
    get_dataloaders)
from utils.config_utils import (
    get_config, 
    get_config_types, 
    verify_config, 
    get_config_dict, 
    get_config_from_file)
from utils.training_utils import (
    parse_args, 
    print_train_configs, 
    set_device, 
    create_progress_bar,
    bits_per_spike,
    set_seeds, 
    create_model_dir)
from utils.model_utils import (
    get_norm,
    get_scheduler,
    get_optimizer)
from utils.logging_utils import (
    start_wandb_sweep, 
    sweep_id_prompt, 
    create_or_get_run_dir,
    launch_wandb_agent)
'''──────────────────────────────── train.py ────────────────────────────────'''
# This file train can train a single NDT-U model, start a hyper parameter sweep,
# or add an agent to a hyperparameter sweep. It has optional arguments that can
# be found by running 'python train.py -h'.

def main():
    # parse CLI arguments
    arg_dict = parse_args()

    # overwrite default config with CLI args and dataset_config
    config = get_config(arg_dict)

    # make sure everything needed is defined
    verify_config(config, arg_dict)

    # prints the config and confirms
    print_train_configs(config, arg_dict)

    # Set GPU used by cuda:0 from config.setup.gpu_idx
    set_device(config, arg_dict)
    device = torch.device('cuda:0')

    # dont print wandb logs
    os.environ['WANDB_SILENT'] = 'true' if config.log.wandb_silent else 'false' 

    # Add an agent to a wandb sweep
    if arg_dict['add']:
        add_sweep_agent(config)

    # Start a wandb sweep
    elif arg_dict['sweep']:
        sweep_id = start_wandb_sweep(config, arg_dict)
        add_sweep_agent(config, sweep_id)
        return
        # Remove wandb sweep folder when completed
        shutil.rmtree(glob('./wandb/*'+sweep_id+'/')[0])
   
    # Run training
    else: run_training(config, device, arg_dict['name'])


def run_training(config, device, name):
    set_seeds(config)

    if config.log.to_wandb:
        # initialize wandb run
        wandb.init(
            project=config.log.wandb_project,
            entity=config.log.wandb_entity,
            config={})

        # if in a sweep, update the config
        if wandb.run.sweep_id != None:
            config = get_config_from_file(f'./wandb/sweep-{wandb.run.sweep_id}/config.yaml') 
            for str_name in wandb.config.keys(): 
                group, key = str_name.split('.') 
                config[group][key] = wandb.config[str_name]  

        # upload final config to wandb
        wandb.config.update(get_config_dict(config), allow_val_change=True)
    
    # create a folder for this run and update the config node
    config.defrost()
    config.dirs.save_dir = create_model_dir(config, name)
    config.freeze()

    # create the dataloaders and return dataset to init model with
    train_dl, val_dl, dataset = get_dataloaders(config, *get_pretraining_data(config))

    # pre-train
    if config.dirs.trained_mdl_path == '':
        model = Transformer(config, dataset)
        # model.load_state_dict(torch.load(config.dirs.trained_mdl_path, strict=False))

# DO I EVEN NEED OLD READINS IF FINE_TUNING MODEL???

    # fine-tune
    else: 
        # get the orinal config used to train the model 
        path = osp.join(osp.dirname(config.dirs.trained_mdl_path), 'config.yaml')
        orig_config = get_config_from_file(path)

        # init model with original config
        model = Transformer(orig_config, dataset)

        # load in parameters
        model.load_state_dict(torch.load(config.dirs.trained_mdl_path))

        # create new readin / readout for new session[s]
        # for session in config.data.sessions:
        #     model.readin[session] = 

    model.to(device)

    train(model, train_dl, val_dl, device)

def add_sweep_agent(config):
    '''Adds an agent to a wandb sweep. If no id is supplied then prompt the user
    to choose from all current sweeps.

    Args:
        config (dict): The config to be used.
        id (str, Optional): The id of the sweep to add this agent to.
    '''
    # if no id given, prompt the user from all current sweeps
    file_list = glob('./wandb/sweep*')
    n_files = len(file_list)

    # if no files were found then exit
    assert n_files != 0, '\nError: No sweeps are currently running.'
    
    # get sweep id, if there is more than one sweep file then prompt user to choose
    id = sweep_id_prompt(file_list) if n_files > 1 else file_list[0].split('/')[-1].split('-')[-1]

    # get original config used to start sweep
    config = get_config_from_file(f'./wandb/sweep-{id}/config.yaml')

    print(f'Adding agent to sweep {id}\n')
    # start agent subprocess
    launch_wandb_agent(config, id)


def train(model, train_dataloader, val_dataloader, fold=''):
    ''' The train function used by all training types (single, sweep).

    Args:
        model (Transformer): The model to be trained.
        train_dataloader (DataLoader): Training set DataLoader.
        val_dataloader (DataLoader): Validation set DataLoader.
        device (torch.device): torch.device object containing the GPU to train
                               on.
    '''
    config = model.config

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # The progress bar shows how much time is remaining and current report
    progress_bar = create_progress_bar(config, model, wandb)

    # Each epoch is a single pass through the entire dataset.
    for epoch in range(config.train.epochs):
        model.train() # turns on dropout

        # step through training data in batches
        for spikes, ho_spikes, sessions in train_dataloader:
            # mask, randomize, and ...
            masked_spikes, labels, loss_mask = model.preprocess_batch(epoch, spikes, ho_spikes)

            # forward pass on masked data
            loss, rates = model(masked_spikes, sessions, labels)

            # log the batch's outputs and inputs and calculate metrics
            model.batch_logger.log(sessions, spikes, ho_spikes, rates, loss, loss_mask)

            # masked timesteps are designated with -100 by model.preprocess_batch()
            msk_loss = loss[loss_mask].mean()

            # backprop only the masked timesteps
            msk_loss.backward()  

            # Clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            # Optimizer step
            optimizer.step()

            # Zero gradients (set to none for speed)
            optimizer.zero_grad(set_to_none=True)

        # Scheduler step
        if scheduler != None:
            scheduler.step()
            model.batch_logger.log_lr(scheduler)

        # Run on the validation set every 'val_interval' epochs
        if val_dataloader != None and epoch % config.train.val_interval == 0:
            model.eval() # turns off dropout

            with torch.no_grad():
                for spikes, ho_spikes, sessions in val_dataloader:
                    labels = spikes.clone()
                    hi_spikes = spikes.clone()
                    
                    if model.has_heldout:
                        labels = torch.cat([labels, ho_spikes], -1)

                    # runinference on validation set
                    loss, rates = model(spikes, sessions, labels)

                    # calculate metrics and log
                    model.batch_logger.log(sessions, spikes, ho_spikes, rates, loss)
                
                # save current state dict 
                torch.save(model.state_dict(), f'{config.dirs.save_dir}/last.pt')

        # average train and val metrics across batches and log
        model.batch_logger.calculate_metrics()

        # save state dict if the model has improved on defined val metric
        if model.batch_logger.has_improved():
            torch.save(model.state_dict(), f'{config.dirs.save_dir}/best.pt')

        # 
        model.batch_logger.update_progress_bar(progress_bar)

        model.batch_logger.log_metrics()

        # check if logged metrics have not improved, if they havent for config.train.es_patience epochs, then stop
        #  TODO FLUSH SCREEN BETTER       
        if model.batch_logger.should_early_stop():
            for i in range(3):
                progress_bar.display(' '*progress_bar.ncols, pos=0) # flush the screen
            progress_bar.close()
            print('\n\n! Early Stopping !\n')
            break
        
        progress_bar.update(1)
    progress_bar.display('', pos=2)
    progress_bar.close()
    print('\nTraining Complete.\n')
    
    # Save the last.pt model
    torch.save(model.state_dict(), f'{config.dirs.save_dir}/best.pt')
    print(f'Saved best & last checkpoints to: \n{config.dirs.save_dir}/')


if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        main()
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')