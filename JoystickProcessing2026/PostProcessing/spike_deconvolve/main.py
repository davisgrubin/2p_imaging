# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:55:19 2026
@author: saminnaji3
"""

import os
import numpy as np
import h5py
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from oasis.functions import deconvolve, estimate_parameters
from modules.ReadResults import read_ROI_label  # Now using your custom ReadResults logic 

# --- Configuration ---
session_path = r'C:\Users\saminnaji3\Downloads\SA18_LG\SA18_20260107'
fs = 30  # Sampling rate
sigma_baseline = 600  # As suggested by Ali for Gaussian F0 calculation

def run_reliable_pipeline(session_path):
    print(f"--- Starting Reliable Pipeline: {os.path.basename(session_path)} ---")
    
    # Setup ops dictionary as required by your ReadResults functions 
    ops = {'save_path0': session_path}
    
    # 1. LOAD RAW DATA
    suite2p_path = os.path.join(session_path, 'suite2p', 'plane0')
    Fraw = np.load(os.path.join(suite2p_path, 'F.npy'))
    Fneu = np.load(os.path.join(suite2p_path, 'Fneu.npy'))
    iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))
    
    # 2. FILTER FOR GOOD ROIs ONLY
    # Using the exact function from your ReadResults.py 
    label_path = os.path.join(session_path, 'ROI_label.h5')
    if os.path.exists(label_path):
        good_indices, _ = read_ROI_label(ops)  # Returns (good_roi, bad_roi) 
        print(f"Loaded {len(good_indices)} ROIs from manual labels.")
    else:
        # Fallback to Suite2p's automated classification
        good_indices = np.where(iscell[:, 0] == 1)[0]
        print(f"Loaded {len(good_indices)} ROIs from Suite2p iscell.")

    # Apply filter to ensure we only process neurons of interest
    Fraw = Fraw[good_indices]
    Fneu = Fneu[good_indices]

    # 3. CALCULATE RELIABLE DFF (Ali's Method)
    # Step A: Neuropil correction (standard 0.7 coefficient)
    F_corr = Fraw - (0.7 * Fneu)
    
    # Step B: Gaussian Baseline Correction for zero-centering
    # Using sigma=600 as requested for the F0 baseline calculation
    F0 = ndimage.gaussian_filter1d(F_corr, sigma_baseline, axis=1)
    dff = (F_corr - F0) / np.maximum(F0, 1e-5)
    
    # 4. OASIS WITH PER-NEURON ESTIMATION
    num_neurons = dff.shape[0]
    inferred_spikes = np.zeros_like(dff)
    denoised_traces = np.zeros_like(dff)
    
    print(f"Inferring spikes for {num_neurons} neurons using auto-estimation...")
    for i in range(num_neurons):
        y = np.asarray(dff[i, :], dtype=np.float64)
        try:
            # Automatic parameter estimation (AR1 model) for individual neuron dynamics
            g, sn = estimate_parameters(y, p=1)
            # Deconvolution (b_nonneg=False handles zero-centered dff)
            c, s, b, g, lam = deconvolve(y, g=g, sn=sn, penalty=1, b_nonneg=False)
            
            inferred_spikes[i, :] = s
            denoised_traces[i, :] = c + b
        except Exception as e:
            print(f"Warning: Neuron {i} failed deconvolution: {e}")

    # 5. SAVE RESULTS
    output_path = os.path.join(session_path, 'qc_results', 'reliable_spikes.h5')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('dff', data=dff)
        f.create_dataset('spikes', data=inferred_spikes)
        f.create_dataset('denoised', data=denoised_traces)
        f.create_dataset('roi_indices', data=good_indices) # Maintain link to original IDs
        
    print(f"Saved reliable results to: {output_path}")
    return dff, inferred_spikes, good_indices


def plot_neuron_diagnostics(dff, spikes, roi_idx, fs=30):

    """Generates ISI and Firing Rate figures for a specific ROI."""

    trace = spikes[roi_idx]

    spike_times = np.where(trace > 1e-4)[0] / fs

    

    if len(spike_times) < 2:

        print(f"Not enough spikes for ROI {roi_idx} diagnostics.")

        return



    # Calculate Metrics

    isi = np.diff(spike_times)

    firing_rate = len(spike_times) / (dff.shape[1] / fs)

    

    # Plotting

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    

    # Left: ISI Histogram (Check for refractory period)

    ax[0].hist(isi, bins=100, range=(0, 2.0), color='teal', alpha=0.7)

    ax[0].set_title(f'ISI Distribution (ROI {roi_idx})\nAvg Rate: {firing_rate:.2f} Hz')

    ax[0].set_xlabel('Time (s)')

    ax[0].set_ylabel('Count')

    

    # Right: Raster + DFF Trace

    time_axis = np.arange(dff.shape[1]) / fs

    ax[1].plot(time_axis, dff[roi_idx], color='gray', alpha=0.5, label='$\Delta F/F$')

    ax[1].scatter(spike_times, [np.max(dff[roi_idx])*1.1]*len(spike_times), 

                  marker='|', color='red', label='Inferred Spikes')

    ax[1].set_title('Trace & Spike Timing')

    ax[1].set_xlabel('Time (s)')

    ax[1].legend()

    

    plt.tight_layout()

    plt.show()

# --- Execution ---

if __name__ == "__main__":

    dff, spikes, good_ids = run_reliable_pipeline(session_path)

    

    # Example diagnostic plot for the first neuron

    plot_neuron_diagnostics(dff, spikes, roi_idx=0, fs=fs)