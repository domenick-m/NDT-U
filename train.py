#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import time
import logging
from glob import glob
from timeit import default_timer as timer
# #────#
import wandb
import torch
import torch.nn as nn
# #────#
from eval import run_evaluation
from model import Transformer
from utils.data_utils import get_dataloaders
from utils.toolkit_utils import get_pretraining_data
from utils.model_utils import get_scheduler, get_optimizer
from utils.logging_utils import start_wandb_sweep, sweep_id_prompt, launch_wandb_agent
from utils.training_utils import (
    set_seeds, 
    parse_args, 
    set_device, 
    create_model_dir,
    print_train_configs, 
    create_progress_bar)
from utils.config_utils import (
    get_config, 
    verify_config, 
    get_config_dict, 
    get_config_from_file,
    update_config_from_sweep)

logging.getLogger('snel_toolkit.datasets.base').setLevel(logging.ERROR)


'''──────────────────────────────── train.py ────────────────────────────────'''
# This file train can train a...


def main():
    '''
    '''
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

    # Add an agent to a wandb sweep
    if arg_dict['add']:
        add_sweep_agent(config)

    # Start a wandb sweep
    elif arg_dict['sweep']:
        sweep_id = start_wandb_sweep(config, arg_dict)
        add_sweep_agent(config, sweep_id)
        # shutil.rmtree(glob('./wandb/*'+sweep_id+'/')[0]) # Remove wandb sweep folder when completed
   
    # Run training
    else: run_training(config, device, arg_dict['name'])


def run_training(config, device, name):
    '''
    '''
    # try to maximize reproducibility (still will vary across machines)
    set_seeds(config)

    # should run be logged to wandb
    if config.log.to_wandb:
        # keep wandb from printing excessive run info
        os.environ['WANDB_SILENT'] = str(config.log.wandb_silent).lower()

        # initialize wandb run, entity defines a user or a group
        wandb.init(project=config.log.wandb_project, entity=config.log.wandb_entity, config={})

        # if in a sweep, update the new parameters received from wandb
        if wandb.run.sweep_id != None:
            config = update_config_from_sweep(config)

        # upload final config to wandb
        wandb.config.update(get_config_dict(config), allow_val_change=True)
    
    # create a folder for this run and update the config node
    config = create_model_dir(config, name)

    # create the dataloaders and return dataset to init model with
    train_dl, val_dl, dataset = get_dataloaders(config, *get_pretraining_data(config))

    # init the model, dataset is used to get data dimensionality
    model = Transformer(config, dataset)

    # always put model on GPU to train
    model.to(device)

    # run training loop
    train(model, train_dl, val_dl)

    # run evaluation on trained model
    run_evaluation(config, model)
    

def add_sweep_agent(config):
    '''
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

    # start agent subprocess
    print(f'Adding agent to sweep {id}\n')
    launch_wandb_agent(config, id)


def train(model, train_dataloader, val_dataloader):
    '''
    '''
    config = model.config

    # use optimizers and schedulers defined in config
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # The progress bar shows how much time is remaining and current report
    progress_bar = create_progress_bar(config, model, wandb)

    # Each epoch is a single pass through the entire dataset.
    for epoch in range(config.train.epochs):
        # turn on dropout
        model.train() 

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
        # if False:
        if val_dataloader != None and epoch % config.train.val_interval == 0:
            # turn off dropout
            model.eval() 

            # dont store gradients for validation
            with torch.no_grad():
                for spikes, ho_spikes, sessions in val_dataloader:
                    labels = spikes.clone()                    

                    # put heldout on labels to calcuate loss over
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

        # update printout of values specified in update_progress_bar 
        model.batch_logger.update_progress_bar(progress_bar)

        # push metrics to wandb or a csv and reset logger
        model.batch_logger.log_metrics()

        # stop training if model hasnt improved in config.train.es_patience epochs
        if model.batch_logger.should_early_stop():
            for i in range(3):
                # flush the console TODO FLUSH SCREEN BETTER   
                progress_bar.display(' '*progress_bar.ncols, pos=i) 
            # close the tqdm progress bar
            progress_bar.close()
            # display message and break from training loop
            print('\n\n! Early Stopping !\n')
            break
        
        # epoch complete, update epoch count on progress bar
        progress_bar.update(1)

    # training complete, flush TODO FLUSH SCREEN BETTER   
    progress_bar.display('', pos=2)

    # close the tqdm progress bar and display message
    progress_bar.close()
    print('\nTraining Complete.\n')
    
    # Save the last.pt state dict and display message
    torch.save(model.state_dict(), f'{config.dirs.save_dir}/last.pt')
    print(f'Saved best & last checkpoints to: \n{config.dirs.save_dir}/')


if __name__ == "__main__":
    import warnings
    warnings.simplefilter('ignore')
    try:
        start_time = time.time()
        main()
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')