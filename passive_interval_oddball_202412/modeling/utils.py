#!/usr/bin/env python3

import numpy as np
from scipy.signal import savgol_filter

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

# normalization into gaussian.
def norm_gauss(data):
    return (data - np.nanmean(data)) / (np.nanstd(data) + 1e-5)

# compute the scale parameters when normalizing data into [0,1].
def get_norm01_params(data):
    a = 1 / (np.nanmax(data) - np.nanmin(data))
    b = - np.nanmin(data) / (np.nanmax(data) - np.nanmin(data))
    return a,b

# compute mean and sem for 3d array data
def get_mean_sem(data, zero_start=False):
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    m = m-m[0] if zero_start else m
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

# compute mean and sem for average df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode='mean'):
    pct = 25
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win = neu_seq[:, l_idx:r_idx].copy()
    neu_win_mean = np.nanmean(neu_win, axis=1)
    # mean values in the window.
    if mode == 'mean':
        neu = neu_win_mean
    # values lower than percentile in the window.
    if mode == 'lower':
        neu = neu_win.reshape(-1)
        neu = neu[neu<np.nanpercentile(neu, pct)]
    # values higher than percentile in the window.
    if mode == 'higher':
        neu = neu_win.reshape(-1)
        neu = neu[neu>np.nanpercentile(neu, pct)]
    # compute mean and sem.
    neu_mean = np.nanmean(neu)
    std = np.nanstd(neu)
    count = np.nansum(~np.isnan(neu_win_mean), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu_mean, neu_sem

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# get derivative.
def compute_derivative(neu_seq, neu_time, win_eval):
    derivative = np.gradient(neu_seq, neu_time)
    return derivative

# smoothing signal.
def smooth_trace(neu_seq, neu_time, win_eval):
    window_size = 15
    polyorder = 3
    if len(neu_seq) < window_size:
        window_size = len(neu_seq) - 1 if len(neu_seq) % 2 == 0 else len(neu_seq) - 2
    if window_size < polyorder:
        polyorder = window_size - 1
    smoothed = savgol_filter(neu_seq, window_size, polyorder)
    return smoothed