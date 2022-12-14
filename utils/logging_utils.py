#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import os.path as osp
import datetime
from statistics import mean
#────#
import yaml
import wandb
#────#
from utils.config_utils import cfg_node_to_dict
import subprocess
import signal
import torch.nn as nn
import torch
import numpy as np
from utils.training_utils import bits_per_spike
'''───────────────────────────── wandb_utils.py _____________________________'''
# This file contains ...

def start_wandb_sweep(config, arg_dict):
    '''Starts a sweep with wandb.sweep and creates a folder with the sweeps
    config in it. Returns the id of the created sweep.

    Args:
        config (dict): A config object.
        arg_dict (CfgNode): Dictionary of the CLI args used when calling train.
    Returns:
        sweep_id (int): The sweep id.
    '''
    log = config.log
    dirs = config.dirs

    cmd = f'wandb sweep -p {log.wandb_project} -e {log.wandb_entity} {dirs.sweep_cfg_path}'
    output = subprocess.getoutput(cmd)
    sweep_id = output.splitlines()[1].split(': ')[2]

    path = f'./wandb/sweep-{sweep_id}/'
    os.makedirs(path)
    with open(path+'/config.yaml', 'w') as yamlfile:
        yaml.dump(cfg_node_to_dict(config), yamlfile)
        
    now = datetime.datetime.now()
    with open(path+'/created.txt', 'w') as cfile:
        cfile.write(now.strftime('%m/%d/%Y, %-I:%M:%S%p'))

    return sweep_id


def sweep_id_prompt(file_list):
    options = 1
    id_list = []
    for file in file_list:
        with open(f'{file}/created.txt', 'r') as cfile:
            timestamp = cfile.readlines()
        sweep_id = file.split('/')[-1].split('-')[-1]
        id_list.append(sweep_id)
        print(str(options)+': '+sweep_id+' - '+timestamp[0])
        options+=1

    try:
        chosen_index = input(f'\nWhich sweep ID should the sweep agent use? (1-{len(id_list)}): ')
        while int(chosen_index) not in range(1,len(id_list)+1):
            chosen_index = input(f'\nPlease enter a number between 1 and {len(id_list)}: ')
    except:
        raise Exception(f'\nError: That is not an integer between 1 and {len(id_list)}.')

    return id_list[int(chosen_index)-1]


def launch_wandb_agent(config, sweep_id):
    my_env = os.environ.copy()

    call = [
        "wandb", "agent", 
        "-p", f"{config.log.wandb_project}", 
        "-e", f"{config.log.wandb_entity}", 
        f'{sweep_id}'
    ]

    with subprocess.Popen(call, env=my_env) as p:
        try: return p.wait()
        except:  
            p.send_signal(signal.SIGINT)
            p.wait()


def create_or_get_run_dir(config, name):
    if name is None: # this means nothing was passed to name CLI arg
        log_local = not config.wandb.log
        # If the reports need to be logged it needs a name
        name = wandb.run.name if log_local else input('\nEnter an ID to save the model with: ')
    
    path = osp.join(config.dirs.model_dir, config.data.type, )
    os.makedirs(path, exist_ok=True)

    if osp.isdir(config.dirs.model_dir):
        file_count = 1
        tmp_path = path
        while osp.isdir(tmp_path):
            tmp_path = path+'_'+str(file_count)
            file_count += 1
        path = tmp_path
        name+=f'_{file_count-1}'
        print(f'\n RENAMING TO: {name}\n')

        
    if config['wandb']['log']: # this means name was passed to CLI args
        wandb.run.name = name
    
    return name


def wandb_cleanup(config):
    '''Deletes the run folder after it's been uploaded.

    Args:
        wandb (wandb): The imported wandb.
    '''
    if wandb.run != None:
        run_id = wandb.run.id
        wandb.finish()
        api = wandb.Api()
        run = api.run(f"{config['wandb']['entity']}/{config['wandb']['project']}/{run_id}")

        max_metrics = [
            'val ho_lt_co_bps', 
            'val ho_co_bps', 
            'val hi_lt_co_bps', 
            'val hi_co_bps'
        ]
        min_metrics = [
            'train nll', 
            'train lt_nll', 
            'val nll', 
            'val lt_nll'
        ]
        avg_metrics = [
            'val ho_lt_co_bps', 
            'val ho_co_bps', 
            'val hi_lt_co_bps', 
            'val hi_co_bps',
            'val nll', 
            'val lt_nll'
        ]

        for metric in max_metrics:
            if metric in run.summary.keys():
                run.summary[f"{metric}_max"] = np.max(run.history()[metric])
            if config['train']['val_type'] == 'cross_val':
                for i in range(config['train']['n_folds']):
                    if f'{metric}_f{i}' in run.summary.keys():
                        run.summary[f"{metric}_max_f{i}"] = np.max(run.history()[f"{metric}_f{i}"])

        for metric in min_metrics:
            if metric in run.summary.keys():
                run.summary[f"{metric}_min"] = np.min(run.history()[metric])
            if config['train']['val_type'] == 'cross_val':
                for i in range(config['train']['n_folds']):
                    if f'{metric}_f{i}' in run.summary.keys():
                        run.summary[f"{metric}_min_f{i}"] = np.min(run.history()[f"{metric}_f{i}"])
        
        # run.summary.update()

        if config['train']['val_type'] == 'cross_val':
            for metric in avg_metrics:
                tmp_list = []
                for i in range(config['train']['n_folds']):
                    if f'{metric}_f{i}' in run.summary.keys():
                        tmp_list.append(run.summary[f'{metric}_f{i}'])
                run.summary[f"{metric}_avg"] = np.mean(tmp_list)
                run.summary[f"{metric}_std"] = np.std(tmp_list)
                if metric in max_metrics:
                    tmp_list = []
                    for i in range(config['train']['n_folds']):
                        if f'{metric}_max_f{i}' in run.summary.keys():
                            tmp_list.append(run.summary[f'{metric}_max_f{i}'])
                    run.summary[f"{metric}_max_avg"] = np.mean(tmp_list)
                    run.summary[f"{metric}_max_std"] = np.std(tmp_list)
                if metric in min_metrics:
                    tmp_list = []
                    for i in range(config['train']['n_folds']):
                        if f'{metric}_min_f{i}' in run.summary.keys():
                            tmp_list.append(run.summary[f'{metric}_min_f{i}'])
                    run.summary[f"{metric}_min_avg"] = np.mean(tmp_list)
                    run.summary[f"{metric}_min_std"] = np.std(tmp_list)

        # if wandb.run != None:
        #     run_id = wandb.run.id
        #     # wandb.finish()
        #     dir_q = get_wandb_dir()
        #     wandb_dir = dir_q if dir_q != None else '.'
        # shutil.rmtree(glob(wandb_dir+'/wandb/*'+run_id+'/')[0])
        run.summary.update()


class BatchedLogger(nn.Module):
    def __init__(self, config, n_heldout):
        super().__init__()
        self.config = config
        self.has_heldout = n_heldout > 0
        self.n_heldout = n_heldout
        self.report_values = []
        self.did_val = False
        self.epoch = 0
        self.best_metric = None
        self.improve_epoch = None
        self.init_data()

    def init_data(self):
        self.lr = 0
        self.did_val = False
        self.results = {}
        self.val_data = {i:[] for i in ['sessions', 'spikes', 'ho_spikes', 'rates', 'loss']}
        self.train_data = {i:[] for i in ['sessions', 'spikes', 'ho_spikes', 'rates', 'loss', 'loss_mask']}

    def log(self, names, spikes, ho_spikes, rates, loss, loss_mask=None):
        # validation
        if loss_mask is None:
            self.val_data['sessions'].append(names)
            self.val_data['spikes'].append(spikes)
            self.val_data['rates'].append(rates)
            self.val_data['loss'].append(loss)
            if self.has_heldout:
                self.val_data['ho_spikes'].append(ho_spikes)
            self.did_val = True

        # training
        else: 
            self.train_data['sessions'].append(names)
            self.train_data['spikes'].append(spikes.detach())
            self.train_data['rates'].append(rates.detach())
            self.train_data['loss'].append(loss.detach())
            self.train_data['loss_mask'].append(loss_mask.detach())
            if self.has_heldout:
                self.train_data['ho_spikes'].append(ho_spikes.detach())

    def log_lr(self, scheduler):
        self.lr = scheduler.optimizer.param_groups[0]['lr']

    def log_to_csv(self):
        with open(f'{self.config.dirs.save_dir}/log.csv', 'a') as f:
            line = '\n'
            for k, v in self.results.items():
                line += f'[{k}: {v}] '
            f.write(line)

    def avg_metrics(self, data_dict, prefix):
        loss = torch.cat(data_dict['loss'], 0)
        rates = torch.cat(data_dict['rates'], 0).exp()
        hi_spikes = torch.cat(data_dict['spikes'], 0)
        if self.has_heldout:
            ho_spikes = torch.cat(data_dict['ho_spikes'], 0)
            spikes = torch.cat([hi_spikes, ho_spikes], -1)
        else: spikes = hi_spikes
        sessions = np.concatenate(data_dict['sessions'], 0)

        # if prefix == 'train':
            # loss_mask = torch.cat(self.train_data['loss_mask'], 0)
            # self.results[f'{prefix}_msk_nll'] = loss[loss_mask].mean()

        self.results[f'{prefix}_nll'] = loss.mean()
        self.results[f'{prefix}_lt_nll'] = loss[:, -1, :].mean()

        if self.has_heldout:
            self.results[f'{prefix}_hi_nll'] = loss[..., :-self.n_heldout].mean()
            self.results[f'{prefix}_hi_lt_nll'] = loss[:, -1, :-self.n_heldout].mean()
            hi_rates = rates[..., :-self.n_heldout]

            self.results[f'{prefix}_ho_nll'] = loss[..., -self.n_heldout:].mean()
            self.results[f'{prefix}_ho_lt_nll'] = loss[:, -1, -self.n_heldout:].mean()
            ho_rates = rates[:,:, -self.n_heldout:]

        for metric in ['bps', 'lt_bps', 'hi_bps', 'hi_lt_bps', 'ho_bps', 'ho_lt_bps']:
            self.results[f'{prefix}_{metric}'] = []

        for session in set(sessions):
            sess_idxs = sessions == session
            # all channels
            bps, lt_bps = bits_per_spike(rates[sess_idxs], spikes[sess_idxs])
            self.results[f'{prefix}_bps'].append(bps)
            self.results[f'{prefix}_lt_bps'].append(lt_bps)
            # self.results[f'{prefix}_{session}_bps'] = bps
            # self.results[f'{prefix}_{session}_lt_bps'] = lt_bps  

            if self.has_heldout:
                # heldin channels
                hi_bps, hi_lt_bps = bits_per_spike(hi_rates[sess_idxs], hi_spikes[sess_idxs])
                self.results[f'{prefix}_hi_bps'].append(hi_bps)
                self.results[f'{prefix}_hi_lt_bps'].append(hi_lt_bps)
                # self.results[f'{prefix}_{session}_hi_bps'] = hi_bps
                # self.results[f'{prefix}_{session}_hi_lt_bps'] = hi_lt_bps        

                # heldout channels
                ho_bps, ho_lt_bps = bits_per_spike(ho_rates[sess_idxs], ho_spikes[sess_idxs])
                self.results[f'{prefix}_ho_bps'].append(ho_bps)
                self.results[f'{prefix}_ho_lt_bps'].append(ho_lt_bps)
                # self.results[f'{prefix}_{session}_ho_bps'] = ho_bps
                # self.results[f'{prefix}_{session}_ho_lt_bps'] = ho_lt_bps
        
        # all channels
        self.results[f'{prefix}_bps'] = mean(self.results[f'{prefix}_bps'])
        self.results[f'{prefix}_lt_bps'] = mean(self.results[f'{prefix}_lt_bps'])

        if self.has_heldout:
            # heldin channels
            self.results[f'{prefix}_hi_bps'] = mean(self.results[f'{prefix}_hi_bps'])
            self.results[f'{prefix}_hi_lt_bps'] = mean(self.results[f'{prefix}_hi_lt_bps'])

            # heldout channels
            self.results[f'{prefix}_ho_bps'] = mean(self.results[f'{prefix}_ho_bps'])
            self.results[f'{prefix}_ho_lt_bps'] = mean(self.results[f'{prefix}_ho_lt_bps'])
        
        self.results['lr'] = self.lr


    def calculate_metrics(self):
        self.avg_metrics(self.train_data, 'train')
        if self.did_val:
            self.avg_metrics(self.val_data, 'val')
        self.epoch += 1

    def log_metrics(self):
        if self.config.log.to_wandb:
            wandb.log(self.results)
        if self.config.log.to_csv:
            self.log_to_csv()
        self.init_data()
        pass

    def has_improved(self):
        metric = 'val_bps'
        improve = 'increase'

        if metric not in self.results:
            return False

        met_val = self.results[metric]

        if (
            self.best_metric == None or
            (improve == 'increase' and met_val > self.best_metric) or 
            (improve == 'decrease' and met_val < self.best_metric)
        ):
            self.best_metric = met_val
            self.improve_epoch = self.epoch
            return True
        
        return False

    def update_progress_bar(self, progress_bar):
        values = ['train_msk_nll', 'train_bps', 'val_bps']

        if not self.report_values:
            self.report_values = ['n/a' for i in values]

        report_str = f'[Epoch: {self.epoch}] '

        for idx, metric in enumerate(values):
            if metric in self.results:
                self.report_values[idx] = f'{self.results[metric]:.3f}'
            report_str += f'[{metric}: {self.report_values[idx]}] '

        progress_bar.display(msg=report_str, pos=0)


    def should_early_stop(self, ):
        if self.config.train.es_patience >= (self.epoch - self.improve_epoch) and self.config.train.early_stopping:
            return True

        return False