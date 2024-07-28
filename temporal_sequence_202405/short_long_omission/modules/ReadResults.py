#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio


# read raw_voltages.h5.

def read_raw_voltages(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw_voltages.h5'),
        'r')
    try:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_hifi = np.array(f['raw']['vol_hifi'])
        vol_img = np.array(f['raw']['vol_img'])
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])
        vol_flir = np.array(f['raw']['vol_flir'])
        vol_pmt = np.array(f['raw']['vol_pmt'])
        vol_led = np.array(f['raw']['vol_led'])
    except:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start_bin'])
        vol_stim_vis = np.array(f['raw']['vol_stim_bin'])
        vol_img = np.array(f['raw']['vol_img_bin'])
        vol_hifi = None
        vol_stim_aud = None
        vol_flir = None
        vol_pmt = None
        vol_led = None
    f.close()
    return [vol_time, vol_start, vol_stim_vis, vol_img, 
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]


# read masks.

def read_masks(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'),'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks_func'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]


# read motion correction offsets.

def read_move_offset(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'move_offset.h5'), 'r')
    xoff = np.array(f['xoff'])
    yoff = np.array(f['yoff'])
    f.close()
    return [xoff, yoff]


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
    neural_trials['time'] = np.array(f['neural_trials']['time'])
    neural_trials['stim'] = np.array(f['neural_trials']['stim'])
    neural_trials['dff'] = np.array(f['neural_trials']['dff'])
    neural_trials['vol_stim'] = np.array(f['neural_trials']['vol_stim'])
    neural_trials['vol_time'] = np.array(f['neural_trials']['vol_time'])
    neural_trials['stim_labels'] = np.array(f['neural_trials']['stim_labels'])
    f.close()
    return neural_trials


# read significance test label results.
def read_significance(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'r')
    significance = {}
    significance['r_normal'] = np.array(f['significance']['r_normal'])
    significance['r_change'] = np.array(f['significance']['r_change'])
    significance['r_oddball'] = np.array(f['significance']['r_oddball'])
    return significance


# read bpod session data.

def read_bpod_mat_data(ops):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d
    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']
    if 'NormalTypes' in raw.keys():
        normal_types = np.array(raw['NormalTypes'])
    else:
        normal_types = np.array(raw['BaselineTypes'])
    fix_jitter_types = np.array(raw['FixJitterTypes'])
    img_seq_label = np.array(raw['ImgSeqLabel'])
    oddball_types = np.array(raw['OddballTypes'])
    if 'OptoTypes' in raw.keys():
        opto_types = np.array(raw['OptoTypes'])
    else:
        opto_types = np.zeros_like(normal_types)
    bpod_sess_data = {
        'normal_types'     : normal_types,
        'fix_jitter_types' : fix_jitter_types,
        'img_seq_label'    : img_seq_label,
        'oddball_types'    : oddball_types,
        'opto_types'       : opto_types,
        }
    return bpod_sess_data
