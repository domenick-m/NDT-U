"""
Classes for datasets used in this project
"""
import os
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import loadmat
from snel_toolkit.datasets.base import BaseDataset

# location of data on shared server
RAW_DATA_DIR = '/snel/share/share/data/bg2/t5_cursor'

# closed-loop block ids
CL_BLOCKS = {
    't5.2021.05.05': 1,
    't5.2021.05.17': 1,
    't5.2021.05.19': 0,
    't5.2021.05.24': 1,
    't5.2021.05.26': 0,
    't5.2021.06.02': 1,
    't5.2021.06.04': 15,
    't5.2021.06.07': 14,
    't5.2021.06.23': 0,
    't5.2021.06.28': 0,
    't5.2021.06.30': 4,
    't5.2021.07.07': 0,
    't5.2021.07.08': 1,
    't5.2021.07.12': 2,
    't5.2021.07.14': 0,
    't5.2021.07.19': 0,
    't5.2021.07.21': 0
}

# labels for continuous fields
CON_FIELDS = {
    # size of the target
    'targetSize': ['size'],
    # size of the cursor
    'cursorSize': ['size'],
    # index of the current block
    'blockNums': ['n'],
    # cursor position
    'cursorPos': ['x', 'y'],
    # target position
    'targetPos': ['x', 'y'],
    # whether the cursor is over the target
    'onTarget': ['state'],
    # decoder's predicted velocity
    'decVel': ['x', 'y'],
    # decoder's predicted click state
    'decClick': ['state'],
}

class T5CursorDataset(BaseDataset):
    """
    Dataset class for data processed with formatWest2DDataset.m in
    github.com/braingate-team/bg-preprocess-t5

    Attributes
    ----------
    data : pandas.DataFrame
        Continuous data
    trial_info : pandas.DataFrame
        Information about each trial in the dataset
    """

    def __init__(self, filename=None):
        """
        Initialize the T5CursorDataset object.
        Parameters
        ----------
        filename : str, optional
            Path to a data file. If specified, load data from this file.
            Defaults to None.
        """
        if filename:
            if filename.endswith('.mat'):
                self.load(filename)
            else:
                raise ValueError("Did not recognize the file extension of "
                                 f"{filename}. Must be .mat.")

    def load(self, filename):
        """
        Load data from a MAT-file.

        Parameters
        ----------
        filename : str
            Path to a MAT-file.
        """
        dataset = loadmat(filename, squeeze_me=True)['dataset']

        # load metadata
        self.subject = 't5'
        self.path = filename
        self.filename = os.path.basename(self.path)
        self.name = os.path.splitext(self.filename)[0]
        self.datetime = datetime.strptime(self.name, r't5.%Y.%m.%d')

        sample_rate = 1e3
        self.bin_width = 1 / sample_rate

        # load continuous data
        labels, signals = {}, {}
        for field in CON_FIELDS:
            labels[field] = CON_FIELDS[field]
            signals[field] = dataset[field].item().squeeze()

        n_samples = signals['cursorPos'].shape[0]
        sample_index = np.arange(n_samples)
        self.time = sample_index / sample_rate
        self.duration = self.time[-1] - self.time[0]

        # load spike data
        # load spike times as a flattened jagged array
        st = dataset['spike_times'].item().squeeze()
        # convert to 0-index
        st -= 1
        # split at these indices to get the spike times for each channel
        st_idx = dataset['spike_times_index'].item().squeeze()
        spike_times = np.split(st, st_idx)[:-1]
        # list of channel IDs
        ch_ids = np.arange(len(spike_times))  # channel IDs

        labels['spikes'] = [f'ch{ch :03d}' for ch in ch_ids]
        signals['spikes'] = self.bin_spikes(spike_times, sample_index)

        # put continuous data into dataframe
        self.data = self.build_continous_df(labels, signals, self.time)

        # load trial information
        self.trial_info = self.build_trial_df(dataset)

    def bin_spikes(self, spike_times, sample_index):
        """
        Bin a sequence of spike times. The bin size is set by self.bin_width.
        Parameters
        ----------
        spike_times : list
            List in which each entry contains the spike times for a unit
        sample_index : np.ndarray
            Index of each sample in the continuous data
        Returns
        -------
        numpy.ndarray
            Array of binned spike counts
        """

        def hist(*args, **kwargs):
            """numpy.histogram with a single return value"""
            return np.histogram(*args, **kwargs)[0]

        # define the edges of the bins
        sample_index = sample_index.astype(int)
        edges = np.append(sample_index, sample_index[-1] + 1)

        # compute bins for each channel
        n_bins = len(edges) - 1
        n_chans = len(spike_times)

        binned_spikes = np.zeros((n_chans, n_bins), dtype=np.uint8)
        binned_spikes[:, :] = Parallel()(
            delayed(hist)(spike_times[ichan], bins=edges)
            for ichan in range(n_chans))

        return binned_spikes.T

    def build_continous_df(self, labels, signals, index):
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
        time_frame = index
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

    def build_trial_df(self, mat_data):
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
        trial_epochs = mat_data['trialEpochs'].item() - 1
        start_ind = trial_epochs[:, 0]
        # subtract 1 from the end_ind to exclude the start of the next trial
        end_ind = trial_epochs[:, 1] - 1

        tinfo = {}
        tinfo['is_successful'] = mat_data['isSuccessful'].item().astype(bool)
        tinfo['target_size'] = mat_data['targetSize'].item()[start_ind]
        tinfo['target_pos_x'] = mat_data['targetPos'].item()[start_ind, 0]
        tinfo['target_pos_y'] = mat_data['targetPos'].item()[start_ind, 1]
        tinfo['block_num'] = mat_data['instructedDelays'].item()
        tinfo['start_time'] = self.time[start_ind]
        tinfo['end_time'] = self.time[end_ind]

        # create trial info dataframe
        tinfo_df = pd.DataFrame(tinfo)

        # compute additional trial attributes
        tinfo_df['is_center_target'] = np.all(
            (tinfo_df['target_pos_x'] == 0, tinfo_df['target_pos_y'] == 0),
            axis=0)

        tinfo_df['reach_angle'] = np.degrees(
            np.arctan2(tinfo_df['target_pos_y'].diff(),
                       tinfo_df['target_pos_x'].diff()))

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
                'blockNums', 'cursorPos', 'cursorSize', 'decClick', 'onTarget',
                'targetPos', 'targetSize'
            ]
        super().resample(*args, **kwargs)

def init_toolkit_dataset(session):
    """
    """
    session_mat_path = osp.join(RAW_DATA_DIR, f'{session}.mat')
    return T5CursorDataset(session_mat_path)