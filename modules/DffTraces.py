#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter


# process dff for opto sessions.

def pmt_led_handler(dff):
    window_size = 3
    dff_filtered = np.apply_along_axis(median_filter, axis=1, arr=dff, size=window_size)
    return dff_filtered


# compute dff from raw fluorescence signals.

def get_dff(
        ops,
        fluo,
        neuropil,
        norm,
        ):
    # correct with neuropil signals.
    dff = fluo.copy() - ops['neucoeff']*neuropil
    # median filtering.
    dff = pmt_led_handler(dff)
    # get baseline.
    f0 = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j,:] = ( dff[j,:] - f0[j,:] ) / f0[j,:]
        if norm:
            # z score.
            dff[j,:] = (dff[j,:] - np.mean(dff[j,:])) / (np.std(dff[j,:]) + 1e-5)
    return dff


# save dff traces results.

def save_dff(ops, dff):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'w')
    f['dff'] = dff
    f.close()


# main function to compute spikings.

def run(ops, norm=True):
    print('===============================================')
    print('=========== dff trace normalization ===========')
    print('===============================================')
    print('Reading fluorescence signals after quality control')
    fluo = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'fluo.npy'),
        allow_pickle=True)
    neuropil = np.load(
        os.path.join(ops['save_path0'], 'qc_results', 'neuropil.npy'),
        allow_pickle=True)
    print('Running baseline subtraction and normalization')
    dff = get_dff(ops, fluo, neuropil, norm)
    print('Results saved')
    save_dff(ops, dff)

