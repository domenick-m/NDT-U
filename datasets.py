#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import sys
import logging
import requests
import os.path as osp
#────#
import h5py
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils import data
from nlb_tools import make_tensors
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    save_to_h5,
    combine_h5,
    h5_to_dict,
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors)

import warnings; warnings.simplefilter('ignore')
'''────────────────────────────── datasets.py ───────────────────────────────'''
# This file contains the class Dataset and the methods for downloading the
# datasets.

'''
   ─────────────────────── DANDI API DATASET STRINGS ────────────────────────
'''
# Hide NaN warnings for test data from nlb_tools
logging.basicConfig(level=logging.ERROR)

# Each dataset has a unique ID that allows us to directly download w/o DandiCLI
api_url = ('https://api.dandiarchive.org/api/assets/', '/download/')
id_dict = dict(
    mc_maze = dict(
        mc_maze_train = '26e85f09-39b7-480f-b337-278a8f034007',
        mc_maze_test = '1bd112a4-5ec5-4033-ac30-d88e70e993d9'),
    mc_rtt = dict(
        mc_rtt_train = '2ae6bf3c-788b-4ece-8c01-4b4a5680b25b',
        mc_rtt_test = '648a7418-98e8-4413-ba97-3772dd325ecc'),
    dmfc_rsg = dict(
        dmfc_rsg_train = '5e92e8db-5212-4607-8b90-3f5a5e319537',
        dmfc_rsg_test = '6718e18a-944f-40f2-8358-d5ff562fb0a0'),
    area2_bump = dict(
        area2_bump_train = 'ded26b6c-418d-43f5-8a37-dfd072c2dbd4',
        area2_bump_test = 'f907b288-a9b4-4c6e-9d66-6812f306e253'),
    mc_maze_small = dict(
        mc_maze_small_train = '7821971e-c6a4-4568-8773-1bfa205c13f8',
        mc_maze_small_test = '544a660e-a173-46f3-985a-60fd67e6ae1f'),
    mc_maze_medium = dict(
        mc_maze_medium_train = '7ef450a8-8684-42e2-8598-cd38ca2b2e50',
        mc_maze_medium_test = 'ac554f8e-7008-4030-9640-6f1cf838a23c'),
    mc_maze_large = dict(
        mc_maze_large_train = 'e67b57b2-e9ad-4d95-b9e3-1262997360dc',
        mc_maze_large_test = 'f42a7bd1-71f4-4972-84a7-2cd1dfbed8a1'))
'''
────────────────────────────────────────────────────────────────────────────────
'''

def get_dataloaders(config, mode):
    ''' Creates Dataset objects and creates the DataLoaders from them.

    Args:
        config (dict): A config object.
        mode (str): ['original', 'random', 'test', 'none'] Which validation set
                    shold be used. Original is the nlb given validation set,
                    random is a random subset of the combined train and
                    validation set, none is no validation set
    Returns:
        train_dataloader (DataLoader): The training set DataLoader.
        (test/val)_dataloader (DataLoader, None): The test or validation set
                                                  DataLoader. In 'none' this is
                                                  None.
    '''
    data_path = config['setup']['data_dir'] + config['setup']['dataset'] + '.h5'
    generator = torch.Generator()
    generator.manual_seed(config['setup']['seed'])

    if mode == 'original':
        train_data = Dataset(config, data_path, 'train')
        val_data = Dataset(config, data_path, 'val')

        val_data.clip_val(train_data.spikes_heldin.max().item() + 3)
        train_dataloader = train_data.get_dataloader(generator, shuffle=True)
        val_dataloader = val_data.get_dataloader(generator, shuffle=False)
        return train_dataloader, val_dataloader

    elif mode == 'random':
        trainval_data = Dataset(config, data_path, 'trainval')

        n_samples = len(trainval_data)
        shuf_gen = torch.Generator()
        shuf_gen.manual_seed(config['setup']['subset_seed'])
        shuffled_inds = torch.randperm(n_samples, generator=shuf_gen)

        split_ind = int((1 - config['train']['val_ratio']) * n_samples)
        train_inds = list(shuffled_inds[:split_ind])
        val_inds = list(shuffled_inds[split_ind:])
        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainval_data, train_inds),
            batch_size=config['train']['batch_size'],
            generator=generator,
            shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainval_data, val_inds),
            batch_size=config['train']['batch_size'],
            generator=generator,
            shuffle=False)
        return train_dataloader, val_dataloader

    elif mode == 'none':
        trainval_data = Dataset(config, data_path, 'trainval')

        trainval_dataloader = trainval_data.get_dataloader(generator, shuffle=True)
        return trainval_dataloader, None

    elif mode == 'test':
        trainval_data = Dataset(config, data_path, 'trainval')
        test_data = Dataset(config, data_path, 'test')

        trainval_dataloader = trainval_data.get_dataloader(generator, shuffle=False)
        test_dataloader = test_data.get_dataloader(generator, shuffle=False)
        return trainval_dataloader, test_dataloader

class Dataset(data.Dataset):
    def __init__(self, config, filename, mode):
        '''init Dataset

        Args:
            config (dict): A config object.
            filename (str): path to the dataset.h5 file.
            mode (str): ['test', 'train' , 'val', 'trainval'] The dataset type.
        '''
        super().__init__()
        with h5py.File(filename, 'r') as h5file:
            h5dict = h5_to_dict(h5file)
            self.config = config

            def set_sizes(self):
                ''' Helper function that assigns the number of samples, trial
                length, forward pass length, number of heldin neurons, and
                number of heldout neurons.
                '''
                self.n_samples = self.spikes_heldin.shape[0]
                self.tr_length = self.spikes_heldin.shape[1]
                self.fp_length = self.spikes_all_fp.shape[1]
                self.full_length = self.tr_length + self.fp_length
                self.n_heldin = self.spikes_heldin.shape[2]
                self.n_heldout = self.spikes_heldout.shape[2]
                self.n_neurons = self.n_heldin + self.n_heldout

            # train_val mode
            if mode == 'train':
                self.spikes_heldin = torch.tensor(h5dict['train_spikes_heldin'].astype(np.float32))
                self.spikes_heldout = torch.tensor(h5dict['train_spikes_heldout'].astype(np.float32))
                self.spikes_all_fp = torch.tensor(h5dict['train_spikes_all_fp'].astype(np.float32))
                set_sizes(self)

            if mode == 'val':
                self.spikes_heldin = torch.tensor(h5dict['eval_spikes_heldin'].astype(np.float32))
                self.spikes_heldout = torch.tensor(h5dict['eval_spikes_heldout'].astype(np.float32))
                self.spikes_all_fp = torch.tensor(h5dict['eval_spikes_all_fp'].astype(np.float32))
                set_sizes(self)

            # trainval mode
            if mode == 'trainval':
                self.spikes_heldin = torch.tensor(h5dict['trainval_spikes_heldin'].astype(np.float32))
                self.spikes_heldout = torch.tensor(h5dict['trainval_spikes_heldout'].astype(np.float32))
                self.spikes_all_fp = torch.tensor(h5dict['trainval_spikes_all_fp'].astype(np.float32))
                set_sizes(self)

            # test mode
            elif mode == 'test':
                heldin = h5dict['test_spikes_heldin']
                samples = h5dict['test_spikes_heldin'].shape[0]
                heldout = np.zeros((
                    samples,
                    h5dict['train_spikes_heldout'].shape[1],
                    h5dict['train_spikes_heldout'].shape[2]
                ))
                forward = np.zeros((
                    samples,
                    h5dict['train_spikes_all_fp'].shape[1],
                    h5dict['train_spikes_all_fp'].shape[2]
                ))
                self.spikes_heldin = torch.tensor(heldin.astype(np.float32))
                self.spikes_heldout = torch.tensor(heldout.astype(np.float32))
                self.spikes_all_fp = torch.tensor(forward.astype(np.float32))
                set_sizes(self)

    def __len__(self):
        '''
        Returns:
            int: The number of samples or trials.
        '''
        return self.spikes_heldin.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index into the batches.
        Returns:
            spikes_heldin (np.ndarray): Spikes held-in.
            spikes_heldout (np.ndarray): Spikes held-out.
            spikes_all_fp (np.ndarray): Spikes forward pass.
        '''
        return (
            self.spikes_heldin[index],
            self.spikes_heldout[index] ,
            self.spikes_all_fp[index]
        )

    def clip_val(self, max):
        self.spikes_heldin = torch.clamp(self.spikes_heldin, max=max)

    def get_dataloader(self, generator, shuffle=True):
        return data.DataLoader(self,
            batch_size=self.config['train']['batch_size'],
            generator=generator,
            shuffle=shuffle)

def verify_dataset(config):
    '''Checks if the correct dataset.h5 file is downloaded, if it isn't then it
    prompts the user if they would like to download it.

    Args:
        config (dict): A config object.
    '''
    data_path = config['setup']['data_dir'] + config['setup']['dataset'] + '.h5'
    if os.path.isfile(data_path):
        return None
    else:
        print('Dataset could not be found in: '+data_path)
        response = input('Would you like to download it? (y/n): ')
        while response != 'y' and response != 'n':
            response = input("Please enter 'y' or 'n': ")
        if response == 'n': exit()
        download_datasets(config['setup']['data_dir'], [config['setup']['dataset']])

def download_datasets(path, datasets = None) -> None:
    ''' Download datasets, combine train/test & prep data, delete old files.

    Adds datasets specified in 'datasets' to download list, if 'datasets' is
    None then download all. Once downloaded, unpack and combine the train / test
    '.nwb' files into a single '.h5' file and store in path. This function WILL
    overwrite dataset files!

    Args:
        path (str): The path where the '.h5' files will be downloaded.
        datasets (list[str]): The datasets to download, if nothing is passed
            then all datasets will be downloaded.
    '''
    print('Downloading Datasets: '+str(datasets if datasets != None else '[ALL]'))
    # Allow '~' in path
    path = osp.expanduser(path)
    # If nothing was passed to datasets, use all datasets from id_dict
    dataset_list = datasets if datasets else id_dict.keys()
    total_files = len(dataset_list) * 2 # train and test for each

    # tqdm is used to display progress bars
    # Need 2 formats because max files download at once is 14, min is 2
    def_format = ('{desc}: {percentage:3.0f}% |{bar}| ', ' / %s files' % total_files)
    fc_3dig = '{n_fmt[0]}{n_fmt[1]}{n_fmt[2]}'
    fc_4dig = '{n_fmt[0]}{n_fmt[1]}{n_fmt[2]}{n_fmt[3]}'
    fc = fc_3dig if total_files < 10 else fc_4dig
    # Make total progress bar
    progress_bar_total = tqdm(
        total=total_files, unit='file',
        unit_scale=True,
        desc='Total',
        bar_format=def_format[0] + fc + def_format[1]
    )

    # dataset_list contains strings of all datasets: 'mc_maze', 'mc_rtt', ...
    for i, dataset in enumerate(dataset_list):
        # id_dict has 2 unique IDs for each dataset, 1 for training and 1 for test
        for dataset_type in id_dict[dataset].keys():
            # Download path is is appended with the new file name
            trial_type = dataset_type.split('_')[-1]
            filepath = path + dataset + '_' + trial_type + '.nwb'
            # Combine the unique IDs and the Dandi api url to download a
            # specific dataset to avoid the use of Dandi CLI
            dataset_id = id_dict[dataset][dataset_type]
            url = api_url[0] + dataset_id + api_url[1]
            response = requests.get(url, stream=True)
            # Get dataset file size
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            # Make individual file progress bar
            progress_bar_file = tqdm(
                total=total_size_in_bytes, unit='iB',
                unit_scale=True, leave=False,
                desc=dataset_type,
                bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt} / {total_fmt}'
                )
            # Start download and update progress bars
            with open(filepath, 'wb') as file:
                for data in response.iter_content(1024): #1024
                    progress_bar_file.update(len(data))
                    progress_bar_total.update(len(data) / total_size_in_bytes)
                    file.write(data)
            # Close file progress bar once file downloaded
            progress_bar_file.close()
        progress_bar_total.refresh()
        # Extract data from nwb files
        progress_bar_unpack = tqdm(
            total=100, unit='%',
            unit_scale=True, leave=False,
            desc='Unpacking '+dataset,
            bar_format='{desc}: {percentage:3.0f}% |{bar}'
            )
        # Turn the two .nwb files in path into one .h5 file and save
        unpack(path, dataset, progress_bar_unpack)
        progress_bar_unpack.close()
    # All files downloaded and unpacked
    progress_bar_total.update(0.001)
    progress_bar_total.close()
    print('\n\nAll Done!')


def unpack(path, dataset, progress_bar):
    ''' Turn train/test '.nwb' files into a single .h5 file and save

    There should only be 2 files in the folder passed to path, 1 train & 1 test
    file for a single dataset. First, combine heldin/out spike forward data into
    '(train/val)_spikes_all_fp' for both the train and test sets of each dataset.
    Then combine the downloaded files, '(dataset)_train.nwb' and
    '(dataset)_test.nwb', to create '(dataset).h5' for each dataset. Store the
    '.h5' file in 'path' and delete the train and test '.nwb' files for each dataset.
    '''
    dataset_obj = NWBDataset(path) # NWB Object
    progress_bar.update(10)
    dataset_obj.resample(5) # bin width of 5 ms
    progress_bar.update(10)
    # ! Training and validation set dictionaries
    train_dict = make_train_input_tensors(
        dataset_obj, dataset_name=dataset,
        trial_split='train', save_file=False,
        include_forward_pred=True
    )
    progress_bar.update(10)
    val_dict = make_eval_input_tensors(
        dataset_obj, dataset_name=dataset,
        trial_split='val', save_file=False
    )
    progress_bar.update(10)
    val_target_dict = make_eval_target_tensors(
        dataset_obj, dataset_name=dataset, save_file=False
    )
    progress_bar.update(10)
    train_dict['train_spikes_all_fp'] = np.concatenate(
        [train_dict.pop('train_spikes_heldin_forward'),
        train_dict.pop('train_spikes_heldout_forward')],
        -1 # Combine neurons
    )
    progress_bar.update(10)
    val_frwd_dict = val_target_dict[dataset]
    val_dict['eval_spikes_all_fp'] = np.concatenate(
        [val_frwd_dict.pop('eval_spikes_heldin_forward'),
        val_frwd_dict.pop('eval_spikes_heldout_forward')],
        -1 # Combine neurons
    )
    progress_bar.update(10)
    # ! Trainval dictionary
    trainval_dict = make_train_input_tensors(
        dataset_obj, dataset_name=dataset,
        trial_split=['train','val'], save_file=False,
        include_forward_pred=True
    )
    trainval_dict['trainval_spikes_heldin'] = trainval_dict.pop('train_spikes_heldin')
    trainval_dict['trainval_spikes_heldout'] = trainval_dict.pop('train_spikes_heldout')
    trainval_dict['trainval_spikes_all_fp'] = np.concatenate(
        [trainval_dict.pop('train_spikes_heldin_forward'),
        trainval_dict.pop('train_spikes_heldout_forward')],
        -1 # Combine neurons
    )
    progress_bar.update(10)
    # ! Test set dictionary
    test_dict = make_eval_input_tensors(
        dataset_obj, dataset_name=dataset,
        trial_split='test', save_file=False
    )
    # Rename eval to test
    test_dict['test_spikes_heldin'] = test_dict.pop('eval_spikes_heldin')
    progress_bar.update(10)
    # Create single dict and save to .h5 file
    h5_file = {**train_dict, **val_dict, **trainval_dict, **test_dict}
    filename = path + dataset + '.h5'
    # Remove older version if it exists
    if osp.isfile(filename): os.remove(filename)
    save_to_h5(h5_file, filename, overwrite=True)
    progress_bar.update(10)
    # Delete the .nwb files
    remove_path = path + dataset + '_'
    for trial_type in ['train', 'test']:
        os.remove(remove_path + trial_type + '.nwb')
