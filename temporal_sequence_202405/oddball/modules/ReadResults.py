#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio


# read ops.npy

def read_ops(list_session_data_path):
    list_ops = []
    for session_data_path in list_session_data_path:
        ops = np.load(
            os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
            allow_pickle=True).item()
        ops['save_path0'] = os.path.join(session_data_path)
        list_ops.append(ops)
    return list_ops


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
        vol_hifi = np.zeros_like(vol_time)
        vol_stim_aud = np.zeros_like(vol_time)
        vol_flir = np.zeros_like(vol_time)
        vol_pmt = np.zeros_like(vol_time)
        vol_led = np.zeros_like(vol_time)
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
    neural_trials['dff'] = np.array(f['neural_trials']['dff'])
    neural_trials['stim_labels'] = np.array(f['neural_trials']['stim_labels'])
    neural_trials['vol_time'] = np.array(f['neural_trials']['vol_time'])
    neural_trials['vol_stim_vis'] = np.array(f['neural_trials']['vol_stim_vis'])
    neural_trials['vol_stim_aud'] = np.array(f['neural_trials']['vol_stim_aud'])
    neural_trials['vol_flir'] = np.array(f['neural_trials']['vol_flir'])
    neural_trials['vol_pmt'] = np.array(f['neural_trials']['vol_pmt'])
    neural_trials['vol_led'] = np.array(f['neural_trials']['vol_led'])
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


# retract all session results.

def read_all(list_ops, sig_tag=None, force_label=None):
    list_labels = []
    list_masks = []
    list_vol = []
    list_dff = []
    list_neural_trials = []
    list_move_offset = []
    list_significance = []
    for ops in list_ops:
        # masks.
        [labels,
         masks,
         mean_func, max_func,
         mean_anat, masks_anat] = read_masks(ops)
        # voltages.
        [vol_time, vol_start, vol_stim_vis, vol_img, 
         vol_hifi, vol_stim_aud, vol_flir,
         vol_pmt, vol_led] = read_raw_voltages(ops)
        # dff.
        dff = read_dff(ops)
        # trials.
        neural_trials = read_neural_trials(ops)
        # movement offset.
        [xoff, yoff] = read_move_offset(ops)
        # significance.
        significance = read_significance(ops)
        if sig_tag == 'all':
            significance['r_normal']  = np.ones_like(significance['r_normal']).astype('bool')
            significance['r_change']  = np.ones_like(significance['r_change']).astype('bool')
            significance['r_oddball'] = np.ones_like(significance['r_oddball']).astype('bool')
        # labels.
        if force_label != None:
            labels = np.ones_like(labels) * force_label
        # append to list.
        list_labels.append(labels)
        list_masks.append(
            [masks,
             mean_func, max_func,
             mean_anat, masks_anat])
        list_vol.append(
            [vol_time, vol_start, vol_stim_vis, vol_img, 
             vol_hifi, vol_stim_aud, vol_flir,
             vol_pmt, vol_led])
        list_dff.append(dff)
        list_neural_trials.append(neural_trials)
        list_move_offset.append([xoff, yoff])
        list_significance.append(significance)
    return [list_labels, list_masks, list_vol, list_dff, 
            list_neural_trials, list_move_offset, list_significance]
        
