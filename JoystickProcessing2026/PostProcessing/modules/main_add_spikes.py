# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:17:27 2026

@author: saminnaji3
"""

import os
import numpy as np
from suite2p.extraction.dcnv import oasis
from modules import DffTraces
import h5py

best_tau = 0.32
fs = 30

def read_dff(ops, folder = False):
    if folder:
        f = h5py.File(os.path.join(ops['save_path0'], 'manual_qc_results\dff.h5'), 'r')
    else:
        f = h5py.File(os.path.join(ops['save_path0'], 'qc_results\dff.h5'), 'r')
        
    if 'dff' in f.keys():
        dff = np.array(f['dff'])
    elif 'name' in f.keys():
        dff = np.array(f['name'])
        
    f.close() 
    return dff

# --- Configuration ---
subject = 'SA15_LG'
subject_id = 'SA15_'
initial_path = r'C:\Users\saminnaji3\Downloads\passive'
output_base = r'C:\Users\saminnaji3\OneDrive - Georgia Institute of Technology\2p imaging\Figures'

# Active session dates
data_dates = ['20251001', '20251002', '20251003', '20251006', '20251007', '20251008', '20251009', '20251010']


# --- Main Execution ---
if __name__ == "__main__":
    final_output_dir = os.path.join(output_base, subject, 'RawTraces', 'Tau_Comparison')

    for data_date in data_dates:
        print(f"\nProcessing Session: {data_date}")
        path = os.path.join(initial_path, subject, subject_id + data_date)
        
        if not os.path.exists(path):
            print(f"   [Skipping] Path not found: {path}")
            continue

        try:
            # Re-read ops to get current neuron count for random sampling
            ops = np.load(os.path.join(path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
            ops['save_path0'] = path
            
            
            if os.path.exists(os.path.join(path, 'qc_results', 'dff.h5')):
                dff = read_dff(ops)
                spikes = oasis(F=dff, batch_size=ops['batch_size'], tau=best_tau, fs=fs)
                DffTraces.save(ops, 'spikes', spikes)
                
            if os.path.isfile(os.path.join(path, 'manual_qc_results', 'dff.h5')):
                dff = read_dff(ops, True)
                spikes = oasis(F=dff, batch_size=ops['batch_size'], tau=best_tau, fs=fs)
                DffTraces.save_manual(ops, 'spikes', spikes)
            
        except Exception as e:
            print(f"   [Error] Failed to process session {data_date}: {e}")