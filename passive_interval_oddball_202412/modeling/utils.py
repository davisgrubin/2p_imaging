#!/usr/bin/env python3

import numpy as np

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

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

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx