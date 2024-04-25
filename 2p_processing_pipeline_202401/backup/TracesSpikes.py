#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter
from suite2p.extraction import oasis


# compute dff from raw fluorescence signals.

def get_dff(
        ops,
        fluo,
        neuropil,
        ):
    # correct with neuropil signals.
    dff = fluo.copy() - ops['neucoeff']*neuropil
    # get baseline.
    f0 = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j,:] = ( dff[j,:] - f0[j,:] ) / f0[j,:]
        # z score.
        dff[j,:] = (dff[j,:] - np.mean(dff[j,:])) / (np.std(dff[j,:]) + 1e-5)
    return dff


# run spike detection on fluorescence signals.

def spike_detect(
        ops,
        dff
        ):
    # oasis for spike detection.
    spikes = oasis(
        F=dff,
        batch_size=ops['batch_size'],
        tau=ops['tau'],
        fs=ops['fs'])
    return spikes


# threshold spikes based on variance.

def thres_spikes(
        ops,
        dff, spikes
        ):
    for i in range(spikes.shape[0]):
        s = spikes[i,:].copy()
        thres = np.mean(dff[i,:]) + ops['spike_thres'] * np.std(dff[i,:])
        s[s<thres] = 0
        spikes[i,:] = s
    return spikes


# main function to compute spikings.

def run(
        ops,
        fluo,
        neuropil,
        ):
    dff = get_dff(ops, fluo, neuropil)
    spikes = spike_detect(ops, dff)
    spikes = thres_spikes(ops, dff, spikes)
    return dff, spikes

