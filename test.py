#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import sys
#────#
import torch
from nlb_tools.make_tensors import save_to_h5
#────#
from datasets import get_dataloaders
from configs.default_config import get_config_from_file
from setup import set_device, set_seeds, setup_runs_folder
'''──────────────────────────────── test.py ─────────────────────────────────'''
# This file takes in the path to a .pt file as an argument and saves
# submission.h5 in save_path/test/run_name where run_name is the name of the
# folder that the .pt file is stored in.

def main():
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
        exit()
    path = sys.argv[1]
    name = path[:path.rindex('/')].split('/')[-1]
    config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')

    set_device(config)
    device = torch.device('cuda:0')

    set_seeds(config)

    model = torch.load(path).to(device)
    model.name = name
    model.eval() # turns on dropout

    trainval_dataloader, test_dataloader = get_dataloaders(config, 'test')

    with torch.no_grad():
        train_rates = []
        # Go through full training set in batches
        for step, (spikes, heldout_spikes, forward_spikes) in enumerate(trainval_dataloader):
            spikes = spikes.to(device)
            ho_spikes = torch.zeros_like(heldout_spikes, device=device)
            fp_spikes = torch.zeros_like(forward_spikes, device=device)
            spikes = torch.cat([torch.cat([spikes, ho_spikes], -1), fp_spikes], 1)

            train_rates.append(model(spikes).cpu())
        train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates

        dataset = trainval_dataloader.dataset
        tr_length = dataset.tr_length # trial length (no forward)
        fp_length = dataset.fp_length # forward pass length
        n_heldin = dataset.n_heldin # number of held in neurons
        n_heldout = dataset.n_heldout # number of held out neurons

        train_rates, _ = torch.split(train_rates, [tr_length, fp_length], 1)
        train_rates_heldin, train_rates_heldout = torch.split(train_rates, [n_heldin, n_heldout], -1)

        eval_rates = []
        # Go through full test set in batches
        for step, (spikes, heldout_spikes, forward_spikes) in enumerate(test_dataloader):
            spikes = spikes.to(device)
            ho_spikes = torch.zeros_like(heldout_spikes, device=device)
            fp_spikes = torch.zeros_like(forward_spikes, device=device)
            spikes = torch.cat([torch.cat([spikes, ho_spikes], -1), fp_spikes], 1)

            eval_rates.append(model(spikes).cpu())
        eval_rates = torch.cat(eval_rates, dim=0).exp() # turn into tensor and use exponential on rates

        eval_rates, eval_rates_forward = torch.split(eval_rates, [tr_length, fp_length], 1)
        eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [n_heldin, n_heldout], -1)
        eval_rates_hi_fp, eval_rates_ho_fp = torch.split(eval_rates_forward, [n_heldin, n_heldout], -1)

        output_dict = {
            config.setup.dataset: {
                'train_rates_heldin': train_rates_heldin.cpu().numpy(),
                'train_rates_heldout': train_rates_heldout.cpu().numpy(),
                'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
                'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
                'eval_rates_heldin_forward': eval_rates_hi_fp.cpu().numpy(),
                'eval_rates_heldout_forward': eval_rates_ho_fp.cpu().numpy()
            }
        }

        save_path = setup_runs_folder(config, model, 'test')
        save_to_h5(output_dict, save_path+'submission.h5', overwrite=True)
        print('Done! Saved to: '+save_path+'submission.h5')


if __name__ == "__main__":
    main()



























#
