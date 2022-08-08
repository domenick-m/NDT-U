#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import time
import shutil
from glob import glob
#────#
import torch
import wandb
import numpy as np
import torch.nn as nn
from nlb_tools.evaluation import bits_per_spike
#────#
from ar_transformer import Transformer
from ar_datasets import verify_dataset, get_dataloaders
from setup import (set_seeds,
                   parse_args,
                   set_device,
                   get_optimizer,
                   get_scheduler,
                   get_wandb_dir,
                   set_sweep_config,
                   setup_runs_folder,
                   print_train_configs,
                   upload_print_results,
                   print_run_get_prog_bar,
                   delete_wandb_run_folder)
from configs.default_config import (get_config,
                                    get_config_dict,
                                    get_wandb_config,
                                    get_config_from_file)
'''──────────────────────────────── train.py ────────────────────────────────'''
# This file train can train a single NDT-U model, start a hyper parameter sweep,
# or add an agent to a hyperparameter sweep. It has optional arguments that can
# be found by running 'python train.py -h'.

def main():
    # Parse arguments
    arg_dict = parse_args(sys.argv[1:])
    # Overwrite default config with CLI args and dataset_config
    config = get_config(arg_dict)
    verify_dataset(config) # makes sure datasets are downloaded, if not then prompt
    print_train_configs(config, arg_dict) # prints the config if starting a single run or sweep
    model_name = arg_dict['--name'] # if no name arg was passed then this is None

    # Set GPU used by cuda:0 from config.setup.gpu_idx
    set_device(config)
    device = torch.device('cuda:0')

    os.environ['WANDB_SILENT'] = config['wandb']['silent'] # dont print wandb logs

    # Add an agent to a wandb sweep
    if '--add' in arg_dict:
        add_sweep_agent(config)
        exit()

    # Run a wandb sweep
    if config['train']['sweep_enabled'] or '--sweep' in arg_dict:
        sweep_id = set_sweep_config(config, arg_dict)
        # The agent function below starts running the run_sweep unitl killed
        wandb.agent(
            sweep_id,
            function=run_sweep,
            count=config['train']['sweep_epochs'] if (
                config['train']['sweep_type'] != 'grid'
            ) else None)
        # Remove wandb sweep folder when completed
        shutil.rmtree(glob('./wandb/*'+sweep_id+'/')[0])

    # Run single train
    else:
        set_seeds(config)
        run_single(config, device, model_name)


def run_single(config, device, name):
    '''Trains a single model according to config using the train function.

    Args:
        config (CfgNode): The config to be used.
        device (torch.device): torch.device object containing the GPU to train
                               on.
        name (str): The model name, used to save model and log model reports.
    '''
    train_dataloader, val_dataloader = get_dataloaders(config, config['train']['val_type'])

    dataset = train_dataloader.dataset # dataset contains all the variables from the Dataset object
    if config['train']['val_type'] == 'random':
        dataset = dataset.dataset # subsets have the object hidden one level further

    if config['wandb']['log']:
        wandb.init(
            dir=get_wandb_dir(),
            project=config['wandb']['project'],
            config=get_config_dict(config))

    if name is None: # this means nothing was passed to name CLI arg
        log_local = config['wandb']['log_local'] and not config['wandb']['log']
        # If the reports need to be logged it needs a name
        name = wandb.run.name if config['wandb']['log'] else (
            input('\nEnter an ID to save the model with: ')) if (
                config['setup']['save_model'] or log_local
            ) else 'unnamed'
    elif config['wandb']['log']: # this means name was passed to CLI args
        wandb.run.name = name

    model = Transformer(config, dataset, name).to(device)
    train(model, train_dataloader, val_dataloader, device)

def run_sweep():
    '''The function that is called by the wandb agent every time it starts a new
    run. Cannot have any arguments!
    '''
    device = torch.device('cuda:0') # no args allowed, create local variable
    wandb.init(
        dir=get_wandb_dir(),
        config=get_config_dict())

    # Make sure wandb uses the same config as the original sweep, CLI args may have been used
    wandb.config.update(
        get_config_dict(get_config_from_file(glob('./wandb/*'+wandb.run.sweep_id+'/')[0]+'config.yaml')),
        allow_val_change=True)
        
    config = get_wandb_config(wandb)
    set_seeds(config)

    notes = ''
    for group in config['wandb']['sweep'].keys():
        for key in config['wandb']['sweep'][group].keys():
            notes += '['+key+': '+str(config[group][key])+'] '
    wandb.run.notes = notes

    print('\nSweep Parameters:\n  '+notes)

    train_dataloader, val_dataloader = get_dataloaders(config, config['train']['val_type'])

    dataset = train_dataloader.dataset # dataset contains all the variables from the Dataset object
    if config['train']['val_type'] == 'random':
        dataset = dataset.dataset # subsets have the object hidden one level further

    model = Transformer(config, dataset, wandb.run.name).to(device)
    train(model, train_dataloader, val_dataloader, device)

def add_sweep_agent(config, id=None):
    '''Adds an agent to a wandb sweep. If no id is supplied then prompt the user
    to choose from all current sweeps.

    Args:
        config (dict): The config to be used.
        id (str, Optional): The id of the sweep to add this agent to.
    '''
    if not id: # prompt the user from all current sweeps
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
            id = file_list[0].split('/')[-1].split('-')[-1]
    print('Adding agent to sweep with ID:', id+'\n')
    # The agent function below starts running the run_sweep until killed
    wandb.agent(
        id,
        function=run_sweep,
        project=config['wandb']['project'],
        count=config['train']['sweep_epochs'] if (
            config['train']['sweep_type'] != 'grid'
        ) else None
    )


def train(model, train_dataloader, val_dataloader, device):
    ''' The train function used by all training types (single, sweep).

    Args:
        model (Transformer): The model to be trained.
        train_dataloader (DataLoader): Training set DataLoader.
        val_dataloader (DataLoader): Validation set DataLoader.
        device (torch.device): torch.device object containing the GPU to train
                               on.
    '''
    scaler = torch.cuda.amp.GradScaler()


    config = model.config
    if config['wandb']['log']:
        wandb.watch(model, log_freq=config['wandb']['log_freq'])

    # If the model needs to log then create a folder for it.
    log_local = config['wandb']['log_local'] and not config['wandb']['log']

    save_path = setup_runs_folder(config, model, 'train') if (
        config['setup']['save_model'] or log_local) else None

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))

    max_co_bps = float('-inf') # used to get the best.pt model
    # The progress bar shows how much time is remaining and current report
    progress_bar = print_run_get_prog_bar(config, model, wandb if (
        config['wandb']['log']) else None)
    # Each epoch is a single pass through the entire dataset.
    for epoch in range(config['train']['epochs']):
        model.train() # sets the dropout to on

        expand_prob = min( # probability to expand mask across multiple timesteps, increases starting at ramp_start epochs
            (epoch - config['train']['ramp_start']) /
            (config['train']['ramp_end'] - config['train']['ramp_start']),
            1)
        # Each step is one batch
        for step, (spikes, heldout_spikes) in enumerate(train_dataloader):
            spikes = spikes.to(device)
            heldout_spikes = heldout_spikes.to(device)
            # forward_spikes = forward_spikes.to(device)
            # preprocess_batch zero masks, randomizes, and sets the labels for masked and unmaksed
            masked_spikes, labels = model.preprocess_batch(
                spikes, # B, T, N
                expand_prob,
                heldout_spikes, # B, T, N
                # forward_spikes, # B, T, N
            )
            # Dont need rates only loss
            with torch.cuda.amp.autocast():
                masked_loss, _ = model(masked_spikes, labels)
                loss = masked_loss.mean()

            if config['wandb']['log']:
                wandb.log({"train loss": loss})
                
            scaler.scale(loss).backward()
            # loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                config['train']['max_grad_norm'])

            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler != None:
                scheduler.step()
        # Run on the validation set every 'val_interval' epochs
        if (epoch % config['train']['val_interval'] == 0 and
            config['train']['val_type'] != 'none'
        ):
            model.eval() # turns off dropout

            all_loss, heldout_loss, forward_loss, heldin_loss = [], [], [], []
            eval_rates, eval_ho_spikes, eval_fw_spikes = [], [], []
            results_dict = {}

            for step, (spikes, heldout_spikes) in enumerate(val_dataloader):
                with torch.no_grad():
                    spikes = spikes.to(device) # B, T, N
                    heldout_spikes = heldout_spikes.to(device) # B, T, N
                    # forward_spikes = forward_spikes.to(device) # B, T, N
                    labels = spikes.clone()
                    labels = torch.cat([labels, heldout_spikes], -1)
                    # labels = torch.cat([labels, forward_spikes], 1)
                    spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes, device=device)], -1)
                    # spikes = torch.cat([spikes, torch.zeros_like(forward_spikes, device=device)], 1)

                    loss, rates = model(spikes, labels)
                    all_loss.append(loss)
                    eval_rates.append(rates)
                    eval_ho_spikes.append(heldout_spikes)
                    # eval_fw_spikes.append(forward_spikes)

                    heldout_masked = labels.clone()
                    heldout_masked[:,:,:-heldout_spikes.size(-1)] = -100
                    ho_loss, ho_rates = model(spikes, heldout_masked)
                    heldout_loss.append(ho_loss)

                    # forward_masked = labels.clone()
                    # forward_masked[:,:-forward_spikes.size(1),:] = -100
                    # fw_loss, fw_rates = model(spikes, forward_masked)
                    # forward_loss.append(fw_loss)

                    heldin_masked = labels.clone()
                    heldin_masked[:,:, -heldout_spikes.size(-1):] = -100
                    # heldin_masked[:, -forward_spikes.size(1):,:] = -100
                    hi_loss = model(spikes, heldin_masked)[0]
                    heldin_loss.append(hi_loss)

            eval_rates = torch.cat(eval_rates, dim=0).exp() # turn into tensor and use exponential on rates
            eval_ho_spikes = torch.cat(eval_ho_spikes, dim=0).cpu().numpy() # turn into tensor
            # eval_fw_spikes = torch.cat(eval_fw_spikes, dim=0).cpu().numpy() # turn into tensor

            all_loss = torch.cat(all_loss, dim=0).cpu().numpy() # send to cpu and convert to numpy
            heldout_loss = torch.cat(heldout_loss, dim=0).cpu().numpy() # send to cpu and convert to numpy
            heldin_loss = torch.cat(heldin_loss, dim=0).cpu().numpy() # send to cpu and convert to numpy
            # forward_loss = torch.cat(forward_loss, dim=0).cpu().numpy() # send to cpu and convert to numpy

            # frwd_splt = [model.tr_length, model.full_length - model.tr_length]
            # eval_rates, eval_rates_forward = torch.split(eval_rates, frwd_splt, 1)
            # eval_rates_forward = eval_rates_forward.cpu().numpy()

            hldt_splt = [model.n_heldin, heldout_spikes.size(-1)]
            eval_rates_heldout = torch.split(eval_rates, hldt_splt, -1)[1].cpu().numpy()

            co_bps = float(bits_per_spike(eval_rates_heldout, eval_ho_spikes))
            # fp_bps = float(bits_per_spike(eval_rates_forward, eval_fw_spikes))
            val_loss = np.mean(all_loss)
            # Save current model if it scores higher than the max_co_bps and is past the save_min_bps
            if (config['setup']['save_model'] and
                co_bps > config['setup']['save_min_bps'] and
                co_bps > max_co_bps
            ):
                torch.save(model, save_path+'best.pt')
                max_co_bps = co_bps

            results_dict = {
                'epoch': str(epoch),
                'val_loss': val_loss,
                'heldout_loss': np.mean(heldout_loss),
                'co_bps': co_bps,
                'forward_loss': np.mean(forward_loss),
                # 'fp_bps': fp_bps,
                'heldin_loss': np.mean(heldin_loss)
            }
            # Update the report above the progress bar and upload to wandb
            upload_print_results(config, results_dict, progress_bar, save_path)
            # If co-bps is lower than the es_min_bps, then stop the training
            if (config['train']['early_stopping'] and
                epoch > config['train']['epochs'] * config['train']['es_chk_pnt'] and
                co_bps < config['train']['es_min_bps']
            ):
                progress_bar.display(' '*progress_bar.ncols, pos=0) # flush the screen
                progress_bar.display(' '*progress_bar.ncols, pos=1)
                progress_bar.display(' '*progress_bar.ncols, pos=2)
                progress_bar.close()
                print('\n\n! Early Stopping !\n')
                break
        wandb.log({'lr':scheduler.optimizer.param_groups[0]['lr'] })

        progress_bar.update(1)
    progress_bar.display('', pos=2)
    progress_bar.close()
    print('\nTraining Complete.\n')
    # Save the last.pt model
    if config['setup']['save_model']:
        torch.save(model, save_path+'last.pt')
        print('Saved best.pt & last.pt to: '+save_path)

    # Remove wandb run file when completed
    delete_wandb_run_folder(wandb)
    # Clear GPU in case in a sweep
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nInterrupted')
        delete_wandb_run_folder(wandb)
