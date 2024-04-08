#!/usr/bin/env python3

import gc
import os
import math
import tifffile
import numpy as np
from tqdm import tqdm
from datetime import datetime
from suite2p.io.tiff import open_tiff


# read the tif files given the filename list.

def read_tif_to_np(
        ops,
        ch_files
        ):
    # no channel data found.
    if len(ch_files) == 0:
        ch_data = np.zeros((1, ops['Lx'], ops['Ly']), dtype='float32')
    # read into array.
    else:
        ch_data = np.empty((0, ops['Lx'], ops['Ly']), dtype='float32')
        for f in tqdm(ch_files):
            data = tifffile.imread(os.path.join(ops['data_path'], f))
            ch_data = np.concatenate((ch_data, data), axis=0)
            # release memory.
            del data
            gc.collect()
    return ch_data

# read neural_trials.h5

def read_neural_trials(ops):
    try:
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
    except:
        print('Fail to read trial data')
        return None


# read mask.h5.

def read_masks(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'masks.h5'),
        'r')
    masks = dict()
    masks['masks'] = np.array(f['masks']['masks'])
    for ch in ['ch1', 'ch2']:
        masks[ch] = dict()
        for k in f['masks'][ch].keys():
            masks[ch][k] = np.array(f['masks'][ch][k])
    f.close()
    return masks
