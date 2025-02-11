#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import percentile_filter

# process dff for opto sessions.
def pmt_led_handler(fluo, dff):
    # find lowest part of values ready for detection.
    fluo_mean = np.percentile(fluo.copy(), 10, axis=0)
    #plt.plot(time_img, fluo_mean)
    # baseline correction.
    fluo_base = percentile_filter(fluo_mean, 90, size=15, mode='nearest')
    fluo_correct = (fluo_mean - fluo_base) / fluo_base
    #plt.plot(time_img, fluo_correct*-200)
    # find threshold based on baseline.
    thres = 2*np.std(fluo_correct[fluo_correct<np.percentile(fluo_correct, 90)])
    #plt.hlines(-thres*-200, time_img[0], time_img[-1], color='black')
    # replace nan with interpolation.
    nan_idx = fluo_correct < -thres
    #plt.plot(time_img, dff[5,:])
    #plt.plot(time_img, nan_idx*500)
    dff[:,nan_idx] = np.nan
    dff = pd.DataFrame(dff)
    dff = dff.interpolate(method='linear', axis=1, limit_direction='both')
    dff = dff.to_numpy()
    return dff

# apply Savitzky-Golay filter to smooth dff.
def smooth_dff(dff):
    window_length=25
    polyorder=5
    dff_filtered = np.apply_along_axis(
        savgol_filter, 1, dff.copy(),
        window_length=window_length, polyorder=polyorder)
    return dff_filtered

# compute dff from raw fluorescence signals.
def get_dff(
        ops,
        dff,
        norm,
        ):
    sig_baseline = 600
    # get baseline.
    f0 = gaussian_filter(dff, [0., sig_baseline])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j,:] = ( dff[j,:] - f0[j,:] ) / f0[j,:]
        if norm:
            # z score.
            dff[j,:] = (dff[j,:] - np.nanmean(dff[j,:])) / (np.nanstd(dff[j,:]) + 1e-5)
    return dff

# save dff traces results.
def save_dff(ops, dff, dff_filtered, fluo):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'w')
    f['dff'] = dff
    f['dff_filtered'] = dff_filtered
    f['fluo'] = fluo
    f.close()

# main function to compute spikings.
def run(ops, norm=True, correct_pmt=False):
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
    dff = fluo.copy() - ops['neucoeff']*neuropil
    print('Running baseline subtraction and normalization')
    dff = get_dff(ops, dff, norm)
    if correct_pmt:
        print('Running PMT/LED fluorescence correction.')
        dff = pmt_led_handler(fluo, dff)
    dff_filtered = smooth_dff(dff)
    print('Results saved')
    save_dff(ops, dff, dff_filtered, fluo)
