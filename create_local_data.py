import os
import numpy as np
import os.path as osp
import scipy.signal as signal
from nlb_tools.make_tensors import save_to_h5
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import train_test_split

def seg_arr(data):
    ''' Segments up a continous datastream split by Nans
    '''
    # shape of data: T (time) x C (channels)
    tmp_list = []
    for channel in data:
        tmp_list.append([channel[seg] for seg in np.ma.clump_unmasked(np.ma.masked_invalid(channel))])

    return np.transpose(np.array(tmp_list), (1, 2, 0))

def chop_data(data, chopsize, overlap, lag_bins):
    ''' Chops data trial by trail (or segment by segment) into overlapping segments.'''
    chopped_data = []
    for trial in data:
        if lag_bins > 0:
            trial = trial[:-lag_bins, :]
        shape = (
            int((trial.shape[0] - overlap) / (chopsize - overlap)),
            chopsize,
            trial.shape[-1],
        )
        strides = (
            trial.strides[0] * (chopsize - overlap),
            trial.strides[0],
            trial.strides[1],
        )

        chopped_trial = np.lib.stride_tricks.as_strided(trial, shape=shape, strides=strides).copy().astype('f')
        chopped_data.append(chopped_trial)

    chopped_data = np.array(chopped_data)

    return chopped_data.reshape((
        chopped_data.shape[0] * chopped_data.shape[1], 
        chopped_data.shape[2], 
        chopped_data.shape[3]
    ))


def get_train_data(smooth_std=50):
    ''' Returns binned heldin spikes (or smoothed spikes) and heldout spikes for training.
    '''
    dataset = NWBDataset('data/mc_rtt_train.nwb')
    dataset.resample(10)

    hi_spikes_trans = dataset.data.spikes.to_numpy().T
    ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

    hi_spike_segments = seg_arr(hi_spikes_trans)[:3] # (4, 16220, 98)
    ho_spike_segments = seg_arr(ho_spikes_trans)[:3] # (4, 16220, 32)

    if smooth_std > 0:
        kern_sd = int(round(smooth_std / dataset.bin_width))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')

        hi_spike_segments = np.apply_along_axis(filt, 1, hi_spike_segments)

    hi_spike_segments = hi_spike_segments.reshape((
        hi_spike_segments.shape[0] * hi_spike_segments.shape[1], 
        hi_spike_segments.shape[2]
    ))

    ho_spike_segments = ho_spike_segments.reshape((
        ho_spike_segments.shape[0] * ho_spike_segments.shape[1], 
        ho_spike_segments.shape[2]
    ))

    return hi_spike_segments, ho_spike_segments

def get_test_data(smooth_std=30, lag=140):
    ''' Returns binned heldin spikes (or smoothed spikes), heldout spikes, and velocity data 
    for training and testing set.
    '''
    dataset = NWBDataset('data/mc_rtt_train.nwb')
    dataset.resample(10)

    vel_trans = dataset.data.finger_vel.to_numpy().T # idx 0 is x; idx 1 is y
    hi_spikes_trans = dataset.data.spikes.to_numpy().T
    ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

    vel_segments = seg_arr(vel_trans) # (4, 16220, 98)
    hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
    ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

    train_hi_spikes, test_hi_spikes = hi_spike_segments[:3], np.expand_dims(hi_spike_segments[3], 0)
    train_ho_spikes, test_ho_spikes = ho_spike_segments[:3], np.expand_dims(ho_spike_segments[3], 0)

    lag_bins = int(round(lag / dataset.bin_width))
    lagged_vel_segments = np.array([seg[lag_bins:] for seg in vel_segments])
    train_vel_segments, test_vel_segments = lagged_vel_segments[:3], np.expand_dims(lagged_vel_segments[3], 0)

    if smooth_std > 0:
        kern_sd = int(round(smooth_std / dataset.bin_width))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')

        train_hi_spikes = np.apply_along_axis(filt, 1, train_hi_spikes)
        test_hi_spikes = np.apply_along_axis(filt, 1, test_hi_spikes)
        test_ho_spikes = np.apply_along_axis(filt, 1, test_ho_spikes)
        train_ho_spikes = np.apply_along_axis(filt, 1, train_ho_spikes)

    train_hi_spikes = train_hi_spikes[:, :-lag_bins, :]
    train_ho_spikes = train_ho_spikes[:, :-lag_bins, :]

    test_hi_spikes = test_hi_spikes[:, :-lag_bins, :]
    test_ho_spikes = test_ho_spikes[:, :-lag_bins, :]

    return (
        train_hi_spikes, train_ho_spikes, train_vel_segments, 
        test_hi_spikes, test_ho_spikes, test_vel_segments
    )

def make_train_data(window=30, overlap=24):
    ''' Creates chopped and binned heldin and heldout spikes for both training 
    and validation sets then stores on disk as an h5 file. 
    '''
    dataset = NWBDataset('data/mc_rtt_train.nwb')
    dataset.resample(10)

    hi_spikes_trans = dataset.data.spikes.to_numpy().T
    ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

    hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
    ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

    train_hi_segments = chop_data(hi_spike_segments[:3], window, overlap, lag_bins=0) # (8097, 30, 98)
    train_ho_segments = chop_data(ho_spike_segments[:3], window, overlap, lag_bins=0) # (8097, 30, 32)

    test_hi_segments = chop_data(np.expand_dims(hi_spike_segments[3], 0), window, 29) # (16191, 30, 98)
    test_ho_segments = chop_data(np.expand_dims(ho_spike_segments[3], 0), window, 29) # (16191, 30, 32)

    train_hi_segments, val_hi_segments = train_test_split(
        train_hi_segments, test_size=0.2, random_state=42) # train: (6477, 30, 98) val: (1620, 30, 98)

    train_ho_segments, val_ho_segments = train_test_split(
        train_ho_segments, test_size=0.2, random_state=42) # train: (6477, 30, 32) val: (1620, 30, 32)

    train_dict = {
        'train_spikes_heldin': train_hi_segments,
        'train_spikes_heldout': train_ho_segments
    }

    val_dict = {
        'val_spikes_heldin': val_hi_segments,
        'val_spikes_heldout': val_ho_segments
    }

    test_dict = {
        'test_spikes_heldin': test_hi_segments,
        'test_spikes_heldout': test_ho_segments
    }

    h5_file = {**train_dict, **val_dict, **test_dict}

    filename = 'data/' + 'mc_rtt_cont_' + str(overlap) + '_train.h5'
    # Remove older version if it exists
    if osp.isfile(filename): os.remove(filename)
    save_to_h5(h5_file, filename, overwrite=True)

def make_test_data(window=30, overlap=24, lag=70, smooth_std=50):
    ''' Creates chopped and binned heldin spikes, heldout spikes, and velocity for training 
    (full) and heldin spikes, heldout spikes, velocity, and smoothed spikes for test set
    then stores on disk as an h5 file. 
    '''
    dataset = NWBDataset('data/mc_rtt_train.nwb')
    dataset.resample(10)

    vel_trans = dataset.data.finger_vel.to_numpy().T # idx 0 is x; idx 1 is y
    hi_spikes_trans = dataset.data.spikes.to_numpy().T
    ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

    vel_segments = seg_arr(vel_trans) # (4, 16220, 2)
    hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
    ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

    lag_bins = int(round(lag / dataset.bin_width))
    lagged_vel_segments = np.array([seg[lag_bins:] for seg in vel_segments])
    train_vel_segments, test_vel_segments = lagged_vel_segments[:3], np.expand_dims(lagged_vel_segments[3], 0)

    train_vel_segments = chop_data(train_vel_segments, window, window - 1, lag_bins=0)[:, -1, :]
    test_vel_segments = chop_data(test_vel_segments, window, window - 1, lag_bins=0)[:, -1, :]

    train_hi_segments = chop_data(hi_spike_segments[:3], window, window - 1, lag_bins) # (8097, 30, 98)
    train_ho_segments = chop_data(ho_spike_segments[:3], window, window - 1, lag_bins) # (8097, 30, 32)

    test_hi_segments = chop_data(np.expand_dims(hi_spike_segments[3], 0), window, window - 1, lag_bins) # (16191, 30, 98)
    test_ho_segments = chop_data(np.expand_dims(ho_spike_segments[3], 0), window, window - 1, lag_bins) # (16191, 30, 32)

    if smooth_std > 0:
        kern_sd = int(round(smooth_std / dataset.bin_width))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')

        test_hi_smth_spikes = np.apply_along_axis(filt, 0, test_hi_segments[:, -1, :])
        test_ho_smth_spikes = np.apply_along_axis(filt, 0, test_ho_segments[:, -1, :])
        train_hi_smth_spikes = np.apply_along_axis(filt, 0, train_hi_segments[:, -1, :])
        train_ho_smth_spikes = np.apply_along_axis(filt, 0, train_ho_segments[:, -1, :])
    
    train_dict = {
        'train_spikes_heldin': train_hi_segments,
        'train_spikes_heldout': train_ho_segments
    }

    test_dict = {
        'test_spikes_heldin': test_hi_segments,
        'test_spikes_heldout': test_ho_segments
    }

    etc_dict = {
        'train_vel_segments': train_vel_segments,
        'test_vel_segments': test_vel_segments,
        'test_hi_smth_spikes': test_hi_smth_spikes,
        'test_ho_smth_spikes': test_ho_smth_spikes,
        'train_hi_smth_spikes': train_hi_smth_spikes,
        'train_ho_smth_spikes': train_ho_smth_spikes
    }

    h5_file = {**train_dict, **test_dict, **etc_dict}

    filename = 'data/' + 'mc_rtt_cont_' + str(overlap) + '_test.h5'
    # Remove older version if it exists
    if osp.isfile(filename): os.remove(filename)
    save_to_h5(h5_file, filename, overwrite=True)
    
smth_std = 50 #ms
lag = 120 #ms
make_test_data(window=30, overlap=24, lag=lag, smooth_std=smth_std)


# ''' CONSTANTS '''
# window = 30 #bins
# overlap = 24 #bins

# smooth_std = 50 #ms

# lag = 120 #ms

# dataset = NWBDataset('data/mc_rtt_train.nwb')
# dataset.resample(10)

# vel_trans = dataset.data.finger_vel.to_numpy().T # idx 0 is x; idx 1 is y
# hi_spikes_trans = dataset.data.spikes.to_numpy().T
# ho_spikes_trans = dataset.data.heldout_spikes.to_numpy().T

# vel_segments = seg_arr(vel_trans) # (4, 16220, 2)
# hi_spike_segments = seg_arr(hi_spikes_trans) # (4, 16220, 98)
# ho_spike_segments = seg_arr(ho_spikes_trans) # (4, 16220, 32)

# train_hi_segments = chop_data(hi_spike_segments[:3], window, overlap) # (8097, 30, 98)
# train_ho_segments = chop_data(ho_spike_segments[:3], window, overlap) # (8097, 30, 32)

# test_hi_segments = chop_data(np.expand_dims(hi_spike_segments[3], 0), window, 29) # (16191, 30, 98)
# test_ho_segments = chop_data(np.expand_dims(ho_spike_segments[3], 0), window, 29) # (16191, 30, 32)

# train_hi_segments, val_hi_segments = train_test_split(
#     train_hi_segments, test_size=0.2, random_state=42) # train: (6477, 30, 98) val: (1620, 30, 98)

# train_ho_segments, val_ho_segments = train_test_split(
#     train_ho_segments, test_size=0.2, random_state=42) # train: (6477, 30, 32) val: (1620, 30, 32)

# train_dict = {
#     'train_spikes_heldin': train_hi_segments,
#     'train_spikes_heldout': train_ho_segments
# }

# val_dict = {
#     'val_spikes_heldin': val_hi_segments,
#     'val_spikes_heldout': val_ho_segments
# }

# test_dict = {
#     'test_spikes_heldin': test_hi_segments,
#     'test_spikes_heldout': test_ho_segments
# }

# h5_file = {**train_dict, **val_dict, **test_dict}

# filename = 'data/' + 'mc_rtt_cont_' + str(overlap) + '.h5'
# # Remove older version if it exists
# if osp.isfile(filename): os.remove(filename)
# save_to_h5(h5_file, filename, overwrite=True)