#!/usr/bin/env python3

import gc
import os
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

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
    downsampling = 1
    file_path = os.path.join(ops['save_path0'], 'raw_voltages.h5')
    with h5py.File(file_path, 'r') as f:
        vol_time = np.array(f['raw']['vol_time'])[::downsampling]
        vol_start = np.array(f['raw']['vol_start'])[::downsampling]
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])[::downsampling]
        vol_hifi = np.array(f['raw']['vol_hifi'])[::downsampling]
        vol_img = np.array(f['raw']['vol_img'])[::downsampling]
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])[::downsampling]
        vol_flir = np.array(f['raw']['vol_flir'])[::downsampling]
        vol_pmt = np.array(f['raw']['vol_pmt'])[::downsampling]
        vol_led = np.array(f['raw']['vol_led'])[::downsampling]
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

# read masks.
def read_masks(ops):
    file_path = os.path.join(ops['save_path0'], 'masks.h5')
    with h5py.File(file_path, 'r') as f:
        labels = np.array(f['labels'])
        masks = np.array(f['masks_func'])
        mean_func = np.array(f['mean_func'])
        max_func = np.array(f['max_func'])
        mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
        masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]

# read motion correction offsets.
def read_move_offset(ops):
    file_path = os.path.join(ops['save_path0'], 'move_offset.h5')
    with h5py.File(file_path, 'r') as f:
        xoff = np.array(f['xoff'])
        yoff = np.array(f['yoff'])
    return [xoff, yoff]

# read dff traces.
def read_dff(ops):
    file_path = os.path.join(ops['save_path0'], 'dff.h5')
    with h5py.File(file_path, 'r') as f:
        dff = np.array(f['dff'])
    return dff

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops):
    file_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    with h5py.File(file_path, 'r') as f:
        neural_trials = dict()
        neural_trials['time'] = np.array(f['neural_trials']['time'], dtype=np.float32)
        neural_trials['dff'] = np.array(f['neural_trials']['dff'], dtype=np.float32)
        neural_trials['stim_labels'] = np.array(f['neural_trials']['stim_labels'], dtype=np.float32)
        neural_trials['vol_time'] = np.array(f['neural_trials']['vol_time'], dtype=np.float32)
        neural_trials['vol_stim_vis'] = np.array(f['neural_trials']['vol_stim_vis'], dtype=np.float32)
        neural_trials['vol_stim_aud'] = np.array(f['neural_trials']['vol_stim_aud'], dtype=np.float32)
        neural_trials['vol_flir'] = np.array(f['neural_trials']['vol_flir'], dtype=np.float32)
        neural_trials['vol_pmt'] = np.array(f['neural_trials']['vol_pmt'], dtype=np.float32)
        neural_trials['vol_led'] = np.array(f['neural_trials']['vol_led'], dtype=np.float32)
    return neural_trials

# read significance test label results.
def read_significance(ops):
    file_path = os.path.join(ops['save_path0'], 'significance.h5')
    with h5py.File(file_path, 'r') as f:
        significance = {}
        significance['r_standard'] = np.array(f['significance']['r_standard'])
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
    standard_types = np.array(raw['StandardTypes'])
    fix_jitter_types = np.array(raw['FixJitterTypes'])
    img_seq_label = np.array(raw['ImgSeqLabel'])
    oddball_types = np.array(raw['OddballTypes'])
    opto_types = np.array(raw['OptoTypes'])
    random_types = np.array(raw['RandomTypes'])
    bpod_sess_data = {
        'standard_types'    : standard_types,
        'fix_jitter_types' : fix_jitter_types,
        'img_seq_label'    : img_seq_label,
        'oddball_types'    : oddball_types,
        'opto_types'       : opto_types,
        'random_types'     : random_types,
        }
    return bpod_sess_data

# get session results for one subject.
def read_subject(list_ops, sig_tag=None, force_label=None):
    list_labels = []
    list_masks = []
    list_vol = []
    list_dff = []
    list_neural_trials = []
    list_move_offset = []
    list_significance = []
    for ops in tqdm(list_ops):
        # masks.
        labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(ops)
        # voltages.
        vol = read_raw_voltages(ops)
        # dff.
        dff = read_dff(ops)
        # trials.
        neural_trials = read_neural_trials(ops)
        # movement offset.
        xoff, yoff = read_move_offset(ops)
        # significance.
        significance = read_significance(ops)
        if sig_tag == 'all':
            significance['r_standard'] = np.ones_like(significance['r_standard']).astype('bool')
            significance['r_change']   = np.ones_like(significance['r_change']).astype('bool')
            significance['r_oddball']  = np.ones_like(significance['r_oddball']).astype('bool')
        # labels.
        if force_label != None:
            labels = np.ones_like(labels) * force_label
        # append to list.
        list_labels.append(labels)
        list_masks.append(
            [masks,
             mean_func, max_func,
             mean_anat, masks_anat])
        list_vol.append(vol)
        list_dff.append(dff)
        list_neural_trials.append(neural_trials)
        list_move_offset.append([xoff, yoff])
        list_significance.append(significance)
        # clear memory usages.
        del labels
        del masks, mean_func, max_func, mean_anat, masks_anat
        del vol
        del dff
        del neural_trials
        del xoff, yoff
        del significance
        gc.collect()
    return [list_labels, list_masks, list_vol, list_dff,
            list_neural_trials, list_move_offset, list_significance]

# get session results for all subject.
def read_all(session_config_list):
    list_labels = []
    list_masks = []
    list_vol = []
    list_dff = []
    list_neural_trials = []
    list_move_offset = []
    list_significance = []
    for i in range(len(session_config_list['list_config'])):
        # read ops for each subject.
        print('Reading subject {}/{}'.format(i+1, len(session_config_list['list_config'])))
        list_session_data_path = [
            os.path.join('results', session_config_list['list_config'][i]['session_folder'], n)
            for n in session_config_list['list_config'][i]['list_session_name']]
        list_ops = read_ops(list_session_data_path)
        # read results for each subject.
        labels, masks, vol, dff, neural_trials, move_offset, significance = read_subject(
             list_ops,
             sig_tag=session_config_list['list_config'][i]['sig_tag'],
             force_label=session_config_list['list_config'][i]['force_label'])
        # append to list.
        list_labels += labels
        list_masks += masks
        list_vol += vol
        list_dff += dff
        list_neural_trials += neural_trials
        list_move_offset += move_offset
        list_significance += significance
        # clear memory usages.
        del labels
        del masks
        del vol
        del dff
        del neural_trials
        del move_offset
        del significance
        gc.collect()
    return [list_labels, list_masks, list_vol, list_dff,
            list_neural_trials, list_move_offset, list_significance]
    
    