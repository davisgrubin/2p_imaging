#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from modules import SpikeDeconv

from .SpikeAnalysis import analyze_spike_traces
# compute dff from raw fluorescence signals.


def get_dff(
        ops,
        fluo,
        neuropil,
        norm,
):
    # correct with neuropil signals.
    dff = fluo.copy() - ops['neucoeff']*neuropil
    # get baseline.
    f0 = gaussian_filter(dff, [0., ops['sig_baseline']])
    for j in range(dff.shape[0]):
        # baseline subtraction.
        dff[j, :] = (dff[j, :] - f0[j, :]) / f0[j, :]
        if norm:
            # z score.
            dff[j, :] = (dff[j, :] - np.mean(dff[j, :])) / \
                (np.std(dff[j, :]) + 1e-5)
    return dff


# save dff traces results.

def save_dff(ops, dff):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'w')
    f['dff'] = dff
    f.close()

# main function to compute spikings.


def run(
        ops,
        norm=True,
        plotting_neurons=[5],
        taus=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]):

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

    # de-convolution code
    tau_spike_dict = {}
    neurons = np.arange(dff.shape[0])
    for tau in taus:
        smoothed, spikes = SpikeDeconv.run(
            ops, dff, oasis_tau=tau, neurons=neurons, plotting_neurons=plotting_neurons)
        tau_spike_dict[tau] = spikes

    analyze_spike_traces(ops, dff, tau_spike_dict,
                         neurons=np.arange(dff.shape[0]))

    print('Results saved')
    save_dff(ops, dff)
    # TODO: write function to save spike traces and convolved traces
