#!/usr/bin/env python3

import os
import h5py
import numpy as np


# read raw_voltages.h5.

def read_raw_voltages(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'raw_voltages.h5'),
            'r')
        vol_time = np.array(f['raw']['vol_time'])
        vol_start_bin = np.array(f['raw']['vol_start_bin'])
        vol_stim_bin = np.array(f['raw']['vol_stim_bin'])
        vol_img_bin = np.array(f['raw']['vol_img_bin'])
        f.close()
        return [vol_time, vol_start_bin, vol_stim_bin, vol_img_bin]
    except:
        raise ValueError('Fail to read voltage data')


# read masks.

def read_masks(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'),'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    ref_img = np.array(f['ref_img'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, ref_img, mean_anat]


# read motion correction offsets.

def read_move_offset(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'move_offset.h5'), 'r')
    xoff = np.array(f['xoff'])
    yoff = np.array(f['yoff'])
    f.close()
    return xoff, yoff


# read dff traces.

def read_dff(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


# read trailized neural traces with stimulus alignment.

def read_neural_trials(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials
