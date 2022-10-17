import os
import sys
import h5py
import torch
import torch.nn as nn
import shutil
import warnings
import wandb
import fileinput
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nlb_tools.make_tensors import h5_to_dict
from plot_utils.plot_rates_vs_spks_indv import plot_rates_vs_spks_indv
from datasets import get_dataloaders
import multiprocessing
import scipy.signal as signal
from nlb_tools.nwb_interface import NWBDataset
import copy

from utils import (get_config,
                   get_config_dict,
                   set_seeds,
                   parse_args)

def seg_arr(data):
    ''' Segments up a continous datastream split by Nans
    '''
    # shape of data: T (time) x C (channels)
    tmp_list = []
    for channel in data:
        tmp_list.append([channel[seg] for seg in np.ma.clump_unmasked(np.ma.masked_invalid(channel))])
    return np.transpose(np.array(tmp_list), (1, 2, 0))

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """equivalent to Pool.map but works with functions inside Class methods"""
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def get_pair_xcorr(
        np_data,
        threshold=0.001,
        zero_chans=False,
        channels=None,
        max_points=None,
        removal="corr",
    ):
        """Calculate the cross-correlations between channels.
        if threshold is set, remove the highly correlated neurons
        from the dataframe.
        signal_type : str
            The signal type to remove correlated channels from.
            Most of the time, it will be 'spikes'.
        threshold : float, optional
            The threshold above which to remove neurons,
            by default None uses no threshold.
        zero_chans : bool, optional
            Whether to zero channels out or remove them
            entirely, by default False.
        channels : list of str, optional
            NOT IMPLEMENTED. Channels to calculate correlation on,
            by default None.
        max_points : int, optional
            The number of points to use when calculating the correlation,
            taken from the beginning of the data, by default None.
        removal : {'corr', 'rate'}, optional
            Whether to remove neurons in the order of the number
            of above-threshold correlations to other neurons or
            of highest firing rate, by default 'corr'. The `rate` option
            is for backwards compatibility with older MATLAB functions.
        """
        assert removal in ["corr", "rate"]

        # todo: add functionality for channels
        if channels is not None:
            raise NotImplementedError

        n_dim = np_data.shape[1]
        pairs = [(i, k) for i in range(n_dim) for k in range(i)]

        def xcorr_func(args):
            i, k = args
            c = np.sum(np_data[:, i] * np_data[:, k]).astype(np.float32)
            if c == 0:
                # print(np.sum(np_data[:, i]))
                # print(np.sum(np_data[:, k]))
                return 0.0
            # normalize
            c /= np.sqrt(np.sum(np_data[:, i] ** 2) * np.sum(np_data[:, k] ** 2))
            return c

        corr_list = parmap(xcorr_func, pairs)
        pair_corr = zip(pairs, corr_list)
        test_arr= []
        chan_names_to_drop = []
        chan_names = [f'{i+1}' for i in range(n_dim)]
        if threshold:
            pair_corr_tmp = list(pair_corr)  # create a copy
            if removal == "corr":
                # sort pairs based on the xcorr values
                pair_corr_tmp.sort(key=lambda x: x[1], reverse=False)
                while pair_corr_tmp:
                    pair, corr = pair_corr_tmp.pop(-1)
                    if corr == 0.0:
                        test_arr.append(pair)
                    if corr > threshold:
                        # get corr for all the other pairs which include the
                        # neurons from this pair
                        c1 = [p[1] for p in pair_corr if pair[0] in p[0]]
                        c2 = [p[1] for p in pair_corr if pair[1] in p[0]]
                        cnt1 = sum(1 for c in c1 if c > threshold)
                        cnt2 = sum(1 for c in c2 if c > threshold)
                        # determine which channel has more number of
                        # highly correlated pairs
                        if cnt1 > cnt2:
                            chan_dropp = pair[0]
                        elif cnt1 < cnt2:
                            chan_dropp = pair[1]
                        else:
                            # if equal, remove the channel with higher mean
                            # correlations
                            if np.mean(c1) > np.mean(c1):
                                chan_dropp = pair[0]
                            else:
                                chan_dropp = pair[1]
                        # remove all the pairs with chan_drop included
                        pair_corr_tmp = [
                            p for p in pair_corr_tmp if chan_dropp not in p[0]
                        ]
                        chan_names_to_drop.append(chan_names[chan_dropp])

        return corr_list, chan_names_to_drop, test_arr


def xcorr(x, y, coeff=True, detrend=False, maxlags=None):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))

    # main xcorr function
    c = np.correlate(x, y, mode='valid')
    if c == 0: return 0.

    if coeff:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly '
                         'positive < %d' % Nx)

    #lags = np.arange(-maxlags, maxlags + 1)
    #c = c[Nx - 1 - maxlags:Nx + maxlags]

    return c

def mp_xcorr( x, y ):
    #x, y = arg
    """ Multiprocessing wrapper fucntion for xcorr """
    # get indices from pair


    '''
    chan_pairs = [ idx_pair for idx_pair in idx_pairs if idx_pair[0] == ichan ]    
    for idx_pair in chan_pairs:
        print( 'INFO: Computing xcorr for ' + str(idx_pair[0]) + ' and ' + str(idx_pair[1]) )
        x = data[ :, idx_pair[0] ]
        y = data[ :, idx_pair[1] ]
        results.append( xcorr( x, y ) )
    '''
    #data = d['data']
    #print( data.shape )
    #print( i )
    #i1 = i[0]
    #i2 = i[1]
    #
    ## get data channels
    #x = data[:,i1]
    #y = data[:,i2]
    x_corr = xcorr(x,y)

    # append xcorr coeff to return list
    return x_corr

def run_xcorr( *args, **kwargs ):
    """ redundant wrapper. can be condensed """
    return mp_run_xcorr( *args, **kwargs )

def mp_run_xcorr( data, n_proc=None ):

    #data = data[ :, :20 ]
    # find idx pairs to compute xcorr
    idx_pairs = []
    # find number of channels in data
    n_chan = data.shape[1]
    print('channels:', n_chan)
    # calculate number of xcorrs to compute
    n_xcorrs = n_chan**2
    for i_xcorr in range( n_xcorrs ):
        idx1 = int(np.ceil( i_xcorr/n_chan ) )
        idx2 = np.mod( i_xcorr, n_chan )
        idx_pair = [ idx1, idx2 ]
        if idx1 < idx2:
            idx_pairs.append( idx_pair )

    results = []
    results = np.full( (len(idx_pairs)), np.nan )
    prev_idx = -1
    for i_xcorr, idx_pair in enumerate( idx_pairs ):
        idx = int(np.ceil( i_xcorr/n_chan ) )
        if idx != prev_idx:
            sys.stdout.write( 'Computing cross correlations for channel ' + str(idx) + '\r' )
            sys.stdout.flush()
            prev_idx = idx
        results[ i_xcorr ] = mp_xcorr( data[ :, idx_pair[0] ], data[ :, idx_pair[1] ] )

    return results, np.array(idx_pairs)

def plot_xcorr_hist( xcorr_data, hist_binwidth=0.0001, title=None , i='error'):
    """
    plot histogram of xcorr counts
    """
    plt.clf()
    plt.Figure()
    plt.subplots(1, 1, figsize=(8, 6))
    data = np.vstack( xcorr_data )
    plt.hist( data, bins=np.arange( min(data), max(data) + hist_binwidth, hist_binwidth ), edgecolor='black')
    # plt.yscale( 'log' )
    plt.xlabel( 'Cross-Correlation' )
    plt.ylabel( 'Counts' )
    if title is not None:
        plt.title( title )
    plt.tight_layout()
    plt.savefig(f'xcorr_test_{i}.png', facecolor='white', transparent=False)
    plt.close()  


def run_xcorr_sweep( dataset_obj, sweep=[ 1, 2, 4, 8, 10 ] ):
    """
    run sweep of cross correlation at different bin widths
    """
    dataset_list = []
    for bin_width in sweep:
        if bin_width == 1:
            hi_spikes_trans = dataset_obj.data.spikes.to_numpy().T
            ho_spikes_trans = dataset_obj.data.heldout_spikes.to_numpy().T

            hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
            ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

            spikes = np.concatenate((
                np.concatenate(hi_spike_segments, axis=0), 
                np.concatenate(ho_spike_segments, axis=0), 
            ), axis=-1)

            dataset_list.append( spikes )
        else:
            print( 'INFO: Copying dataset' )
            dataset = copy.deepcopy( dataset_obj )
            print( 'INFO: Resampling dataset to ' + str(bin_width) + ' ms' )
            dataset.resample(bin_width)

            hi_spikes_trans = dataset.data.spikes.to_numpy().T
            ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

            hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
            ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

            spikes = np.concatenate((
                np.concatenate(hi_spike_segments, axis=0), 
                np.concatenate(ho_spike_segments, axis=0), 
            ), axis=-1)

            dataset_list.append( spikes )

    xcorr_list = []
    idx_pairs_list = []
    for bin_width, spikes in zip( sweep, dataset_list ):
        title = ' '.join( [ 'mc_rtt', 'X-Corr', str(bin_width) + 'ms' ] )
        print( 'INFO: Geting spiking data' )
        print( 'INFO: Running xcorr at ' + str( bin_width ) )
        x_corr, idx_pairs = run_xcorr( spikes )
        xcorr_list.append( x_corr )
        idx_pairs_list.append( idx_pairs )
        plot_xcorr_hist( x_corr, title=title, i=bin_width)

    return xcorr_list, idx_pairs_list

arg_dict = parse_args(sys.argv[1:])
config = get_config(arg_dict)

filepath = f'{config["setup"]["data_dir"]}{config["setup"]["dataset"]}.nwb'

dataset_obj = NWBDataset(filepath) # NWB Object
dataset_obj.resample(500)


t_data = dataset_obj.make_trial_data(allow_nans=False, allow_overlap=False)

hi_spikes_trans = t_data.spikes
ho_spikes_trans = t_data.heldout_spikes

# hi_spikes_trans = dataset_obj.data.spikes.to_numpy().T
# ho_spikes_trans = dataset_obj.data.heldout_spikes.to_numpy().T

# hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
# ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

# spikes = np.concatenate((
#     np.concatenate(hi_spike_segments, axis=0), 
#     np.concatenate(ho_spike_segments, axis=0), 
# ), axis=-1)

spikes = np.concatenate((
    hi_spikes_trans, 
    ho_spikes_trans, 
), axis=-1)

print(spikes.shape)

pair_corr, chan_names_to_drop, test = get_pair_xcorr(spikes)

print(pair_corr)

title = 'mc_rtt XCorr 1ms'
hist_binwidth = 0.0001

# test = [int(i) for i in chan_names_to_drop ]
# test2 = [i for i in range(1, 131)]
# test3 = np.setdiff1d(test2, test)
# print(test3)

plt.clf()
plt.Figure()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
data = np.vstack( pair_corr )
plt.hist( data, bins=np.arange( min(data), max(data) + hist_binwidth, hist_binwidth ), edgecolor='black')
plt.yscale( 'log' )
plt.xlabel( 'Cross-Correlation' )
plt.ylabel( 'Counts' )
if title is not None:
    plt.title( title )
plt.tight_layout()
plt.savefig(f'non correlated channels.png', facecolor='white', transparent=False)
plt.close()  

# plt.clf()
# plt.Figure()
# fig, ax = plt.subplots(len(test3), 1, figsize=(15, len(test3)*2))
# # data = np.vstack( pair_corr )
# for idx, i in enumerate(ax):
#     i.plot(spikes[:10000, idx])
# # ax1.plot(spikes[:10000, 0])
# # ax2.plot(spikes[:10000, 1])
# # plt.hist( data, bins=np.arange( min(data), max(data) + hist_binwidth, hist_binwidth ), edgecolor='black')
# # plt.yscale( 'log' )
# plt.xlabel( 'Cross-Correlation' )
# plt.ylabel( 'Counts' )
# if title is not None:
#     plt.title( title )
# plt.tight_layout()
# plt.savefig(f'non correlated channels.png', facecolor='white', transparent=False)
# plt.close()  

# print('xcorr_list: \n', pair_corr, '\n\nidx_pairs_list:\n', idx_pairs_list)