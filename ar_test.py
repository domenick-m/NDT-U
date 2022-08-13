#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import sys
#────#
import torch
from nlb_tools.make_tensors import save_to_h5
#────#
from tqdm import tqdm
import numpy as np
from datasets import get_dataloaders
from configs.default_config import get_config_from_file
from setup import set_device, set_seeds, setup_runs_folder
from ar_datasets import chop
'''──────────────────────────────── test.py ─────────────────────────────────'''
# This file takes in the path to a .pt file as an argument and saves
# submission.h5 in save_path/test/run_name where run_name is the name of the
# folder that the .pt file is stored in.

def chop_and_infer(func,
                   data,
                   seq_len=30,
                   stride=1,
                   batch_size=64,
                   output_dim=None,
                   func_kw={}):
    if stride > seq_len:
        raise ValueError(
            "Stride must be less then or equal to the sequence length")
    device = torch.device('cuda:0')
    data_len, data_dim = data.shape[0], data.shape[1]
    output_dim = data_dim if output_dim is None else output_dim

    batch = np.zeros((batch_size, seq_len, data_dim), dtype=np.float64)
    output = np.zeros((data_len, output_dim), dtype=np.float64)
    olap = seq_len - stride

    n_seqs = (data_len - seq_len) // stride + 1
    n_batches = np.ceil(n_seqs / batch_size).astype(int)

    i_seq = 0  # index of the current sequence
    for i_batch in tqdm(range(n_batches)):
        n_seqs_batch = 0  # number of sequences in this batch
        # chop
        start_ind_batch = i_seq * stride
        for i_seq_in_batch in range(batch_size):
            if i_seq < n_seqs:
                start_ind = i_seq * stride
                batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
                                                   seq_len]
                i_seq += 1
                n_seqs_batch += 1
        end_ind_batch = start_ind + seq_len
        # infer
        batch_out = func(torch.Tensor(batch).to(device), **func_kw)[:n_seqs_batch]
        # print(batch_out)
        n_samples = n_seqs_batch * stride

        # merge
        if start_ind_batch == 0:  # fill in the start of the sequence
            output[:olap, :] = batch_out[0, :olap, :].detach().cpu().numpy()
        
        # print(batch_out[0, :olap, :].detach().numpy())

        # print(output[:olap, :])

        out_idx_start = start_ind_batch + olap
        out_idx_end = end_ind_batch
        out_slice = np.s_[out_idx_start:out_idx_end]
        output[out_slice, :] = batch_out[:, olap:, :].reshape(
            n_samples, output_dim).detach().cpu().numpy()

    return output


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
    # trainval_dataloader, test_dataloader = get_dataloaders(config, 'original')

    with torch.no_grad():
        train_rates = []
        # Go through full training set in batches
        for step, (spikes, heldout_spikes, forward_spikes) in enumerate(trainval_dataloader):
            spikes = spikes.to(device)
            ho_spikes = torch.zeros_like(heldout_spikes, device=device)
            fp_spikes = torch.zeros_like(forward_spikes, device=device)
            spikes = torch.cat([spikes, ho_spikes], -1)
            # spikes = torch.cat([torch.cat([spikes, ho_spikes], -1), fp_spikes], 1)
            for idx, i in enumerate(spikes):
                test = chop_and_infer(model, i.cpu().numpy(), config['train']['chop_size'])
                test = torch.cat([torch.unsqueeze(torch.Tensor(test), dim=0), torch.zeros((1, 40, 130))], 1)
                train_rates.append(test)
                # else:
                #     train_rates.append(model(i).cpu())

            # train_rates.append(model(spikes).cpu())
        train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates
        print(train_rates.shape)

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
            spikes = torch.cat([spikes, ho_spikes], -1)
            for i in spikes:
                test = chop_and_infer(model, i.cpu().numpy(), config['train']['chop_size'])
                test = torch.cat([torch.unsqueeze(torch.Tensor(test), dim=0), torch.zeros((1, 40, 130))], 1)
                eval_rates.append(test)
                # test = chop_and_infer(model, i.cpu().numpy(), config['train']['chop_size'])
                # eval_rates.append(torch.unsqueeze(torch.Tensor(test), dim=0))
            # eval_rates.append(model(spikes).cpu())
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
