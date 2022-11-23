#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import os.path as osp
import datetime
#────#
import yaml
import wandb
#────#
from utils.config_utils import cfg_node_to_dict
import subprocess
import signal
'''───────────────────────────── wandb_utils.py _____________________________'''
# This file contains ...


def get_swept_params(sweep_cfg_dir):
    with open(sweep_cfg_dir, 'rb') as yamlf:
        yaml_dict = yaml.load(yamlf)
        param_list = [i.split('.')[1] for i in yaml_dict['parameters'].keys()]
        value_list = [str(i) for i in yaml_dict['parameters'].values()]
    return param_list, value_list

def start_wandb_sweep(config, arg_dict):
    '''Starts a sweep with wandb.sweep and creates a folder with the sweeps
    config in it. Returns the id of the created sweep.

    Args:
        config (dict): A config object.
        arg_dict (CfgNode): Dictionary of the CLI args used when calling train.
    Returns:
        sweep_id (int): The sweep id.
    '''
    with open(config.dirs.sweep_cfg_path, 'rb') as yamlf:
        yaml_dict = yaml.load(yamlf)

    yaml_dict['project'] = config['wandb']['project']
    yaml_dict['entity'] = config['wandb']['entity']

    sweep_id = wandb.sweep(yaml_dict, project=config['wandb']['project'])

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
    if config.train.gpu != -1:
        my_env["CUDA_VISIBLE_DEVICES"] = str(config.train.gpu)

    call = [
        "wandb", "agent", 
        "-p", f"{config.wandb.project}", 
        "-e", f"{config.wandb.entity}", 
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
        project = wandb.run.project if log_local else 'Local'
    

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