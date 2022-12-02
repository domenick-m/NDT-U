#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import os
import os.path as osp
from datetime import datetime
#────#
import numpy as np
import pandas as pd
from scipy.io import loadmat
from snel_toolkit.datasets.base import BaseDataset


# location of data on shared server
RAW_DATA_DIR = '/snel/share/data/braingate/T11/T11PianoSessions'

class T11PianoDataset(BaseDataset):
    """
    

    Attributes
    ----------
    data : pandas.DataFrame
        Continuous data
    trial_info : pandas.DataFrame
        Information about each trial in the dataset
    """

    def __init__(self, session_dir=None):
        """

        Parameters
        ----------
        session_dir : str, optional
            Path to a sessions directory. If specified, load data from all 
            blocks in the session.
            Defaults to None.
        """
        if session_dir is not None:
            assert osp.isdir(session_dir), \
                f'Session directory could not be found at: {session_dir}'

            # store .mat arrays for each block seperately
            self.block_mats = {}

            # loop thorugh everything in the session directory 
            for block in os.listdir(session_dir):
                block_path = osp.join(session_dir, block)
                # make sure block_path is a folder, not extractInfo.mat 
                if osp.isdir(block_path):
                    block_id = int(block.split('-')[1])
                    # store data, info, and task arrays
                    self.block_mats[block_id] = self.load(block_path)

            self.session_dir = session_dir
            
            # combine blocks then build trial_info and continuous data
            self.process_data()


    def load(self, block_dir):
        """
        Load data from a MAT-file.

        Parameters
        ----------
        filename : str
            Path to the data MAT-file.
        """
        # get paths to .mat files
        data_mat_path = osp.join(block_dir, 'data.mat')
        task_mat_path = osp.join(block_dir, 'task.mat')

        # make sure all mat files exist
        assert osp.exists(data_mat_path), \
            f'Could not find data.mat file at: {data_mat_path}'
        assert osp.exists(task_mat_path), \
            f'Could not find task.mat file at: {task_mat_path}'

        # load mat files into np arrays
        data_arr = loadmat(data_mat_path, squeeze_me=True)
        task_arr = loadmat(task_mat_path, squeeze_me=True)

        return {'data':data_arr, 'task':task_arr}


    def process_data(self):
        # load metadata
        self.subject = 't11'
        self.blocks = self.block_mats.keys()
        self.name = osp.basename(self.session_dir)
        self.datetime = datetime.strptime(self.name, r'%Y-%m-%d')

        self.bin_width = 0.02 # 20 ms in seconds
        
        # labels for continuous fields
        labels = {
            # block number
            'block_num': ['n'],
            # condition id, 0 is no cue and 1+ is cued digit
            'cond_id': ['n'],
        }

        # get number of channels and samples from each block
        n_samples = {}
        for block, arr_dict in self.block_mats.items():
            spks_shape = arr_dict['data']['ncTX'].shape
            n_samples[block] = spks_shape[0]
            # number of channels should be the same for all blocks
            n_channels = spks_shape[1]

        # assign labels to the channels
        ch_labels = [f'ch{i:03d}' for i in np.arange(n_channels)]
        labels['spikes'] = ch_labels
        labels['spike_power'] = ch_labels

        mats = self.block_mats.values()

        block_num = [np.repeat(i, n_samples[i]) for i in self.block_mats.keys()]
        cond_id = [i['data']['labels'] for i in mats]
        spikes = [i['data']['ncTX'] * self.bin_width for i in mats]
        spike_power = [i['data']['spikePower'] for i in mats]

        # combine blocks for continuous signals
        signals = {
            'block_num': np.concatenate(block_num, 0),
            'cond_id': np.concatenate(cond_id, 0),
            'spikes': np.concatenate(spikes, 0),
            'spike_power': np.concatenate(spike_power, 0),
        }

        # create indices for the dataframe
        sample_index = np.arange(sum(n_samples.values()))
        self.time = sample_index * self.bin_width
        self.duration = self.time[-1] - self.time[0]

        # put continuous data into dataframe
        self.data = self.build_continous_df(labels, signals, self.time)

        # load trial information
        self.trial_info = self.build_trial_df(n_samples)


    def build_continous_df(self, labels, signals, time_frame):
        """
        Build a Pandas DataFrame from continuous signals.
        Parameters
        ----------
        labels : dict
            Names to use for each column of features. Each key is the name of
            a signal. Each value is a list containing a name for each channel
            in the signal.
        signals : dict
            Signals to use for the DataFrame. Each key is the name of
            a signal (must match labels). Each value is an array of shape
            (n_samples, n_features) containing the data for that signal.
        index : pandas.Index or array-like
            Index to use for resulting DataFrame

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame containing the data
        """
        frames = []
        for signal_type, channels in labels.items():
            # create tuples that name each column as multiIndex
            midx_tuples = [(signal_type, channel) for channel in channels]
            midx = pd.MultiIndex.from_tuples(midx_tuples,
                                             names=('signal_type', 'channel'))

            # create a dataframe for each signal_type
            signal = signals[signal_type]
            signal_type_data = pd.DataFrame(signal,
                                            index=time_frame,
                                            columns=midx)

            signal_type_data.index.name = 'clock_time'
            frames.append(signal_type_data.copy())

        # concatenate data into one continuous dataframe with timedelta indices
        data = pd.concat(frames, axis=1)
        data.index = pd.to_timedelta(data.index, unit='s')
        return data

    def merge_to_df(self, labels, signals):
        """
        Add additional signals to the self.data DataFrame
        Parameters
        ----------
        labels : dict
            Names to use for each column of features. Each key is the name of
            a signal. Each value is a list containing a name for each channel
            in the signal.
        signals : dict
            Signals to use for the DataFrame. Each key is the name of
            a signal (must match labels). Each value is an array of shape
            (n_samples, n_features) containing the data for that signal.
        """
        data = self.build_continous_df(labels=labels,
                                       signals=signals,
                                       index=self.data.index)
        self.data = self.data.join(data)

    def build_trial_df(self, n_samples):
        """
        Create a DataFrame with information on the trials in this dataset

        Parameters
        ----------
        mat_data : numpy.ndarray
            Contents of the `dataset` variable in the MAT-file

        Returns
        -------
        pandas.DataFrame
            DataFrame containing information for each trial
        """
        # store .mat arrays for all blocks
        mats = self.block_mats.values()

        # start/stop indices are relative to block, add offset
        curr_idx = 0
        all_indices = []
        for idx, zipped in enumerate(zip(mats, n_samples.values())):
            mat_dict, samples = zipped
            # start/stop indices are 1 indexed because of matlab
            start_stops = mat_dict['task']['startStops'] - 1
            all_indices.append(start_stops + curr_idx)
            curr_idx += samples

        trial_epochs = np.concatenate(all_indices, 0)
        start_ind = trial_epochs[:, 0]
        # subtract 1 from the end_ind to exclude the start of the next trial
        end_ind = trial_epochs[:, 1] - 1

        trial_types = {
            'Intertrial': 'IT',
            'Cued task CL': 'CL',
            'Cued task calibration': 'CL',
            'Cued task OL': 'OL'
        }
        trial_type = []
        for mat_dict in mats:
            for trial in mat_dict['task']['trialType']:
                trial_type.append(trial_types[trial])

        block_num = [i['task']['blockNumber'] for i in mats]
        cond_id = [i['task']['labels'] for i in mats]
        dec_id = [i['task']['decodes'] for i in mats]
        decode_success = \
            [i['task']['decodes'] == i['task']['labels'] for i in mats]

        tinfo = {
            'block_num': np.concatenate(block_num, 0),
            'trial_type': trial_type,
            'start_time': self.time[start_ind],
            'end_time': self.time[end_ind],
            'cond_id': np.concatenate(cond_id, 0),
            'dec_id': np.concatenate(dec_id, 0),
            'decode_success': np.concatenate(decode_success, 0),
        }

        # create trial info dataframe
        tinfo_df = pd.DataFrame(tinfo)

        # convert time values to timedelta
        for field in ['start_time', 'end_time']:
            tinfo_df[field] = pd.to_timedelta(tinfo_df[field], unit='s')

        return tinfo_df

    def make_trial_data(self, *args, **kwargs):
        """
        See snel_toolkit.datasets.base.BaseDataset.make_trial_data for full
        documentation
        """
        tdata = super().make_trial_data(*args, **kwargs)
        tdata.sort_index(axis=1, inplace=True)
        return tdata

    def resample(self, *args, **kwargs):
        """
        Recompute self.trial_info using new indices after resampling the
        dataset
        """
        if 'discrete_fields' not in kwargs:
            # skip antialiasing on fields that contains discrete data
            kwargs['discrete_fields'] = [
                'block_num', 'cond_id'
            ]
        super().resample(*args, **kwargs)

def init_toolkit_dataset(session):
    """
    """
    session_mat_path = osp.join(RAW_DATA_DIR, f'{session}.mat')
    return T11PianoDataset(session_mat_path)