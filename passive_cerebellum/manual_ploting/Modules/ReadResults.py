#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.signal import savgol_filter


# create a numpy memmap from an h5py dataset.
def create_memmap(data, dtype, mmap_path):
    memmap_arr = np.array(data, dtype=dtype)
    return memmap_arr

# create folder for h5 data.
def get_memmap_path(base_path, h5_file_name):
    mm_folder_name, _ = os.path.splitext(h5_file_name)
    mm_path = os.path.join(base_path, 'memmap', mm_folder_name)
    file_path = os.path.join(base_path, h5_file_name)
    return mm_path, file_path

# read masks.
def read_masks(ops, manual = False):
    if manual:
        f = h5py.File(os.path.join(ops['save_path0'], 'manual_qc_results/masks.h5'), 'r')
    else:
        f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'), 'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks_func'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    masks_anat_corrected = np.array(
        f['masks_anat_corrected']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected]

def read_ROI_label(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'ROI_label.h5'), 'r')
    good_roi = np.array(f['good_roi'])
    bad_roi = np.array(f['bad_roi'])
    f.close()
    return good_roi, bad_roi


# read dff traces.
def read_dff(ops, manual = False):
    if manual:
        f = h5py.File(os.path.join(ops['save_path0'], 'manual_qc_results/dff.h5'), 'r')
    else:
        f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
        
    if 'dff' in f.keys():
        dff = np.array(f['dff'])
    elif 'name' in f.keys():
        dff = np.array(f['name'])
        
    f.close() 
    return dff

# read dff traces.
def read_spikes(ops, manual = False):
    if manual:
        f = h5py.File(os.path.join(ops['save_path0'], 'manual_qc_results/spikes.h5'), 'r')
    else:
        f = h5py.File(os.path.join(ops['save_path0'], 'spikes.h5'), 'r')
    
    if 'spikes' in f.keys():
        spikes = np.array(f['spikes'])
    elif 'name' in f.keys():
        spikes = np.array(f['name'])
    f.close() 
    return spikes

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops):
    # read h5 file.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'session_data', 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])     
    f.close()
    return neural_trials

def read_neural_data(ops, smooth = 0):
    mm_path, file_path = get_memmap_path(ops['save_path0'], 'neural_trials.h5')

    with h5py.File(file_path, 'r') as f:
        neural_trials = {}
        group = f['neural_trials']

        for key in group.keys():
            data = np.array(group[key])

            if key == 'dff' and smooth:
                data = np.apply_along_axis(
                    savgol_filter, 1, data,
                    window_length=11,
                    polyorder=2
                )

            neural_trials[key] = create_memmap(
                data,
                'float32',
                os.path.join(mm_path, f'{key}.mmap')
            )

    return neural_trials

def read_cluster_labels(ops, name = 'neuron_cluster_ids'):
    
    f = h5py.File(os.path.join(ops['save_path0'], name+ '.h5'), 'r')
    labels = np.array(f['labels'])
    f.close()
    return labels