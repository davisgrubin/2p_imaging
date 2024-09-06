#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.ndimage import percentile_filter
from suite2p.extraction import oasis


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


# use spike deconvolution to smooth dff.
def smooth_dff(ops, dff):
    tau = 5
    win = 5
    # oasis for spike detection.
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=tau,
        fs=ops['fs'])
    # convolution with gaussian kernel.
    dff_s = gaussian_filter(spikes, [0., win])
    # normalization.
    for j in range(dff.shape[0]):
        dff_s[j,:] = dff_s[j,:]/ (np.nanstd(dff_s[j,:]) + 1e-5)
    dff_f = np.apply_along_axis(gaussian_filter, axis=1, arr=dff, size=25)
    return dff_f

    
# compute dff from raw fluorescence signals.
def get_dff(
        ops,
        dff,
        norm,
        ):
    # get baseline.
    f0 = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j,:] = ( dff[j,:] - f0[j,:] ) / f0[j,:]
        if norm:
            # z score.
            dff[j,:] = (dff[j,:] - np.nanmean(dff[j,:])) / (np.nanstd(dff[j,:]) + 1e-5)
    return dff


# save dff traces results.

def save_dff(ops, dff, fluo):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'w')
    f['dff'] = dff
    f['fluo'] = fluo
    f.close()


# main function to compute spikings.

def run(ops, norm=True, smooth=False, correct_pmt=False):
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
    if smooth:
        print('Running trace smoothing')
        dff = smooth_dff(ops, dff)
    if correct_pmt:
        print('Running PMT/LED fluorescence correction.')
        dff = pmt_led_handler(fluo, dff)
    print('Results saved')
    save_dff(ops, dff, fluo)

