#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import numpy as np
import scipy.signal as signal
'''─────────────────────────────── general.py ───────────────────────────────'''
# General data utilities.

# def chop(data, seq_len, overlap):
#     ''' TODO
#     '''
#     shape = (int((data.shape[0] - overlap) / (seq_len - overlap)), seq_len, data.shape[-1])
#     strides = (data.strides[0] * (seq_len - overlap), data.strides[0], data.strides[1])
#     return np.lib.stride_tricks.as_strided(data, shape, strides).copy().astype('f')

# def smooth_spikes(data, gauss_width, bin_width, causal):
#     ''' TODO
#     '''
#     kern_sd = int(gauss_width / bin_width)
#     window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
#     if causal: 
#         window[len(window) // 2:] = 0
#     window /= np.sum(window)
#     filt = lambda x: np.convolve(x, window, 'same')
#     return np.apply_along_axis(filt, 0, data)
