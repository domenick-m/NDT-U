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
        sweep_id = set_sweep_config(config, arg_dict)
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
    '''Trains a single model according to config using the train function.

    Args:
        config (CfgNode): The config to be used.
        device (torch.device): torch.device object containing the GPU to train
                               on.
        name (str): The model name, used to save model and log model reports.
    '''
    set_seeds(config)
    cross_val_enabled = (config['train']['val_type'] == 'cross_val')

    if config['wandb']['log']:
        wandb.init(
            dir=get_wandb_dir(),
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=get_config_dict(config))
    
    if wandb.run.sweep_id != None:
        wandb.config.update(
            get_config_dict(get_config_from_file(glob('./wandb/*'+wandb.run.sweep_id+'/')[0]+'config.yaml')), 
            allow_val_change=True
        )
        config = CN(get_wandb_config())

    name = get_run_name(config, name)

    train_dataloader, val_dataloader = get_dataloaders(
        config, 
        'train_val'
        # 'cross_val' if cross_val_enabled else 'train_val'
    )

    if cross_val_enabled:
        for idx, train_sub_dl, val_sub_dl in val_dataloader:
            run_name = f'{name}_f{idx}'
            dataset_sub = train_sub_dl.dataset.dataset
            model = Transformer(config, dataset_sub, run_name, device).to(device)
            train(model, train_sub_dl, val_sub_dl, device, f'_f{idx}')
        val_dataloader = None
        dataset = train_dataloader.dataset 
    else:
        dataset = train_dataloader.dataset.dataset 

    model = Transformer(config, dataset, name, device).to(device)
    train(model, train_dataloader, val_dataloader, device)
    test_func(config, model)

    # wandb_cleanup()
    torch.cuda.empty_cache()


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


def train(model, train_dataloader, val_dataloader, device, fold=''):
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
    counter = 9

    # If the model needs to log then create a folder for it.
    log_local = config['wandb']['log_local'] and not config['wandb']['log']

    save_path = setup_runs_folder(config, model, 'train') if (
        config['setup']['save_model'] or log_local
    ) else None

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Init values to track val improvement
    comp_met_val = metric_comparison(config)
    es_counter = 0

    # The progress bar shows how much time is remaining and current report
    progress_bar = print_run_get_prog_bar(config, model, wandb if (
        config['wandb']['log']) else None)

    report = ['-', '-', '-', '-']

    # Each epoch is a single pass through the entire dataset.
    for epoch in range(config['train']['epochs']):
        model.train() # turns on dropout

        # Probability to expand mask across multiple timesteps, increases starting at ramp_start epochs
        expand_prob = min(1,
            (epoch - config['train']['ramp_start']) / (config['train']['ramp_end'] - config['train']['ramp_start'])
        )

        # Create new results dict every epoch, either upload to wandb or write to csv
        results_dict = {}

        # Each step is one batch
        for batch in train_dataloader:
            if model.has_heldout:
                spikes, heldout_spikes, names = batch
            else:
                spikes, names = batch
                heldout_spikes = None

            # preprocess_batch zero masks, randomizes, and sets the labels for masked and unmaksed
            masked_spikes, labels = model.preprocess_batch(
                spikes, # B, T, N
                expand_prob,
                heldout_spikes, # B, T, N
            )

            # with torch.cuda.amp.autocast():
            loss, _, _, _, _, _, _ = model(masked_spikes, names, labels)
            # with torch.cuda.amp.autocast():
            #     loss, _ = model(masked_spikes, names, labels)

            lt_nll = loss[:, -1, :].mean()
            msk_nll = loss[labels != -100].mean()
            
            results_dict['epochs'] = epoch
            results_dict['train nll'] = loss.mean()
            results_dict['train lt_nll'] = lt_nll
            results_dict['train msk_nll'] = msk_nll

            if config['wandb']['log'] and epoch % config['train']['val_interval'] != 0: 
                wandb.log({k+fold if k != 'epochs' else 'epochs': v for k, v in results_dict.items()})

            # Backprop loss
            if config['train']['lt_loss_only']:
                lt_nll.backward()  
            else: 
                msk_nll.backward()  
            # scaler.scale(lt_nll if config['train']['lt_loss_only'] else msk_nll).backward()

            # Clip gradient
            nn.utils.clip_grad_norm_(model.parameters(), config['train']['max_grad_norm'])

            # Optimizer step
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if scheduler != None:
            scheduler.step()
            results_dict['lr'] = scheduler.optimizer.param_groups[0]['lr']

        # Run on the validation set every 'val_interval' epochs
        if val_dataloader != None and epoch % config['train']['val_interval'] == 0:
            model.eval() # turns off dropout

            with torch.no_grad():
                if model.has_heldout:
                    spikes, ho_spikes, names = val_dataloader 
                else:
                    spikes, names = val_dataloader
                    ho_spikes = None

                labels = spikes.clone()
                hi_spikes = spikes.clone()
                
                if model.has_heldout:
                    labels = torch.cat([labels, ho_spikes], -1)

                # with torch.cuda.amp.autocast():
                loss, rates, factors_saved, pe_saved, pe_factors_saved, output_saved, readouts = model(spikes, names, labels)
                counter += 1
                if counter == 10:
                    counter = 0
                    # plt.pcolormesh(pe_factors_saved.T.cpu().numpy(), cmap = 'Reds' )
                    # plt.tight_layout()
                    # plt.savefig(f"images/{model.name}_pos_facs.png")
                    # plt.close()

                    # plt.pcolormesh(factors_saved.T.cpu().numpy(), cmap = 'Reds' )
                    # plt.tight_layout()
                    # plt.savefig(f"images/{model.name}_facs.png")
                    # plt.close()

                    # # plt.pcolormesh(labels[0,:,:].T.cpu().numpy(), cmap = 'Reds' )
                    # # plt.tight_layout()
                    # # plt.savefig(f"images/{model.name}_spks.png")
                    # # plt.close()

                    # plt.pcolormesh(rates[0,:,:].T.exp().cpu().numpy(), cmap = 'Reds' )
                    # plt.tight_layout()
                    # plt.savefig(f"images/{model.name}_rates.png")
                    # plt.close()

                    # plt.pcolormesh(output_saved.T.exp().cpu().numpy(), cmap = 'Reds' )
                    # plt.tight_layout()
                    # plt.savefig(f"images/{model.name}_output.png")
                    # plt.close()
                    
                    # pos_emb = pe_saved.T
                    # plt.pcolormesh(pos_emb.cpu().numpy(), cmap = 'Reds' )
                    # plt.tight_layout()
                    # plt.savefig(f"images/{model.name}_pos_emb.png")
                    # plt.close()

                    # pic_dict = {
                    #     "rates": wandb.Image(f"images/{model.name}_rates.png"), 
                    #     # "spikes": wandb.Image(f"images/{model.name}_spks.png"),
                    #     "factors": wandb.Image(f"images/{model.name}_facs.png"),
                    #     "pos factors": wandb.Image(f"images/{model.name}_pos_facs.png"),
                    #     "pos emb": wandb.Image(f"images/{model.name}_pos_emb.png"),
                    #     "output": wandb.Image(f"images/{model.name}_output.png")
                    # }
                    # # for idx, readout in enumerate(readouts):
                    # #     plt.pcolormesh(readouts[readout].weight.cpu().numpy(), cmap = 'Reds' )
                    # #     plt.tight_layout()
                    # #     plt.savefig(f"images/{model.name}_readout_{readout}.png")
                    # #     plt.close()
                    # #     pic_dict[f"readout_{readout}"] = wandb.Image(f"images/{model.name}_readout_{readout}.png") 

                    # #     plt.pcolormesh(readouts[readout].bias.unsqueeze(1).cpu().numpy(), cmap = 'Reds' )
                    # #     plt.tight_layout()
                    # #     plt.savefig(f"images/{model.name}_readout_{readout}_bias.png")
                    # #     plt.close()
                    # #     pic_dict[f"readout_{readout}_bias"] = wandb.Image(f"images/{model.name}_readout_{readout}_bias.png") 

                    # wandb.log(pic_dict)


                if model.has_heldout:
                    n_heldout = ho_spikes.shape[-1]

                    hi_nll = loss[:,:, :-n_heldout]
                    ho_nll = loss[:,:, -n_heldout:]
                    hi_lt_nll = loss[:,-1, :-n_heldout]
                    ho_lt_nll = loss[:,-1, -n_heldout:]

                    hi_rates = rates[:,:, :-n_heldout].exp()
                    ho_rates = rates[:,:, -n_heldout:].exp()
                    hi_co_bps = bits_per_spike(hi_rates, hi_spikes)
                    ho_co_bps = bits_per_spike(ho_rates, ho_spikes)
                    hi_lt_co_bps = bits_per_spike(hi_rates[:, -1:, :], hi_spikes[:, -1:, :])
                    ho_lt_co_bps = bits_per_spike(ho_rates[:, -1:, :], ho_spikes[:, -1:, :])
                
                    results_dict['val hi_nll'] = hi_nll.mean()
                    results_dict['val ho_nll'] = ho_nll.mean()
                    results_dict['val hi_lt_nll'] = hi_lt_nll.mean()
                    results_dict['val ho_lt_nll'] = ho_lt_nll.mean()
                    results_dict['val hi_co_bps'] = hi_co_bps
                    results_dict['val ho_co_bps'] = ho_co_bps
                    results_dict['val hi_lt_co_bps'] = hi_lt_co_bps
                    results_dict['val ho_lt_co_bps'] = ho_lt_co_bps

                results_dict['val nll'] = loss.mean()
                results_dict['val lt_nll'] = loss[:, -1:, :].mean()
                
                improved, ovrw_val, comp_metric = metric_comparison(config, comp_met_val, results_dict)
                
                if config['setup']['save_model']:
                    torch.save(model.state_dict(), save_path+'last_sd.pt')
                    
                if improved:
                    comp_met_val = ovrw_val
                    es_counter = 0
                    if config['setup']['save_model']:
                        torch.save(model, save_path + f'best_{comp_metric}.pt')
                else: es_counter += 1

                if config['setup']['save_model']:
                    torch.save(model, save_path+'last.pt')

        # Update the report above the progress bar and upload to wandb
        upload_print_results(config, report, results_dict, progress_bar, save_path, fold)

        # Early Stopping
        if es_counter >= config['train']['es_patience'] and config['train']['early_stopping']:
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
    if config['setup']['save_model']:
        torch.save(model, save_path+'last.pt')
        print('Saved best & last checkpoints to: '+save_path)


if __name__ == "__main__":
    try:
        # main()
        import time
        start_time = time.time()
        main()
        print("\n--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        print('\n\nInterrupted')
        wandb_cleanup()
