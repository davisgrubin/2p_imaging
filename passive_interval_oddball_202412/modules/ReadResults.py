#!/usr/bin/env python3

import gc
import os
import copy
import h5py
import shutil
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from scipy.signal import savgol_filter

# filter configuration list for data reading.
def filter_session_config_list(session_config_list, target_sess):
    sub_session_config_list = copy.deepcopy(session_config_list)
    for si in range(len(sub_session_config_list['list_config'])):
        sub_session_config_list['list_config'][si]['list_session_name'] = {
            k: v for k, v in session_config_list['list_config'][si]['list_session_name'].items()
            if v == target_sess}
    return sub_session_config_list

# create a numpy memmap from an h5py dataset.
def create_memmap(data, dtype, mmap_path):
    memmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=data.shape)
    memmap_arr[:] = data[...]
    return memmap_arr

# create folder for h5 data.
def get_memmap_path(ops, h5_file_name):
    mm_folder_name, _ = os.path.splitext(h5_file_name)
    if not os.path.exists(os.path.join(ops['save_path0'], 'memmap', mm_folder_name)):
        os.makedirs(os.path.join(ops['save_path0'], 'memmap', mm_folder_name))
    mm_path = os.path.join(ops['save_path0'], 'memmap', mm_folder_name)
    file_path = os.path.join(ops['save_path0'], h5_file_name)
    return mm_path, file_path

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
    mm_path, file_path = get_memmap_path(ops, 'raw_voltages.h5')
    with h5py.File(file_path, 'r') as f:
        vol_time     = create_memmap(f['raw']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        vol_start    = create_memmap(f['raw']['vol_start'],    'int8',    os.path.join(mm_path, 'vol_start.mmap'))
        vol_stim_vis = create_memmap(f['raw']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        vol_hifi     = create_memmap(f['raw']['vol_hifi'],     'int8',    os.path.join(mm_path, 'vol_hifi.mmap'))
        vol_img      = create_memmap(f['raw']['vol_img'],      'int8',    os.path.join(mm_path, 'vol_img.mmap'))
        vol_stim_aud = create_memmap(f['raw']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        vol_flir     = create_memmap(f['raw']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        vol_pmt      = create_memmap(f['raw']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        vol_led      = create_memmap(f['raw']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

# read dff traces.
def read_dff(ops):
    mm_path, file_path = get_memmap_path(ops, 'dff.h5')
    with h5py.File(file_path, 'r') as f:
        dff = create_memmap(f['dff'], 'float32', os.path.join(mm_path, 'dff.mmap'))
    return dff

# read camera dlc results.
def read_camera(ops, z_score=True):
    try:
        h5_file_name = [f for f in os.listdir(ops['save_path0']) if 'camera' in f][0]
        mm_path, file_path = get_memmap_path(ops, h5_file_name)
        with h5py.File(file_path, 'r') as f:
            camera_time  = np.array(f['camera_dlc']['camera_time'], dtype='float32')*1000
            camera_time -= camera_time[0]
            camera_pupil = np.array(f['camera_dlc']['pupil'], dtype='float32')
            if z_score:
                camera_pupil = (camera_pupil - np.nanmean(camera_pupil)) / (1e-10 + np.nanstd(camera_pupil))
    except:
        camera_time  = np.array([np.nan], dtype='float32')
        camera_pupil = np.array([np.nan], dtype='float32')
    return [camera_time, camera_pupil]

# read masks.
def read_masks(ops):
    mm_path, file_path = get_memmap_path(ops, 'masks.h5')
    with h5py.File(file_path, 'r') as f:
        labels     = create_memmap(f['labels'],     'int8',    os.path.join(mm_path, 'labels.mmap'))
        masks      = create_memmap(f['masks_func'], 'float32', os.path.join(mm_path, 'masks_func.mmap'))
        mean_func  = create_memmap(f['mean_func'],  'float32', os.path.join(mm_path, 'mean_func.mmap'))
        max_func   = create_memmap(f['max_func'],   'float32', os.path.join(mm_path, 'max_func.mmap'))
        mean_anat  = create_memmap(f['mean_anat'],  'float32', os.path.join(mm_path, 'mean_anat.mmap')) if ops['nchannels'] == 2 else None
        masks_anat = create_memmap(f['masks_anat'], 'float32', os.path.join(mm_path, 'masks_anat.mmap')) if ops['nchannels'] == 2 else None
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]

# read motion correction offsets.
def read_move_offset(ops):
    mm_path, file_path = get_memmap_path(ops, 'move_offset.h5')
    with h5py.File(file_path, 'r') as f:
        xoff = create_memmap(f['xoff'], 'int8', os.path.join(mm_path, 'xoff.mmap'))
        yoff = create_memmap(f['yoff'], 'int8', os.path.join(mm_path, 'yoff.mmap'))
    return [xoff, yoff]

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops, smooth):
    mm_path, file_path = get_memmap_path(ops, 'neural_trials.h5')
    with h5py.File(file_path, 'r') as f:
        neural_trials = dict()
        dff = np.array(f['neural_trials']['dff'])
        if smooth:
            window_length=9
            polyorder=3
            dff = np.apply_along_axis(
                savgol_filter, 1, dff,
                window_length=window_length,
                polyorder=polyorder)
        else: pass
        neural_trials['dff']          = create_memmap(dff,                                'float32', os.path.join(mm_path, 'dff.mmap'))
        neural_trials['time']         = create_memmap(f['neural_trials']['time'],         'float32', os.path.join(mm_path, 'time.mmap'))
        neural_trials['stim_labels']  = create_memmap(f['neural_trials']['stim_labels'],  'int32'  , os.path.join(mm_path, 'stim_labels.mmap'))
        neural_trials['vol_time']     = create_memmap(f['neural_trials']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        neural_trials['vol_stim_vis'] = create_memmap(f['neural_trials']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        neural_trials['vol_stim_aud'] = create_memmap(f['neural_trials']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        neural_trials['vol_flir']     = create_memmap(f['neural_trials']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        neural_trials['vol_pmt']      = create_memmap(f['neural_trials']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        neural_trials['vol_led']      = create_memmap(f['neural_trials']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
        neural_trials['camera_time']  = create_memmap(f['neural_trials']['camera_time'],  'float32', os.path.join(mm_path, 'camera_time.mmap'))
        neural_trials['camera_pupil'] = create_memmap(f['neural_trials']['camera_pupil'], 'float32', os.path.join(mm_path, 'camera_pupil.mmap'))
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
def read_subject(list_ops, sig_tag, force_label, smooth):
    list_labels = []
    list_masks = []
    list_neural_trials = []
    list_move_offset = []
    list_significance = []
    for ops in tqdm(list_ops):
        if not os.path.exists(os.path.join(ops['save_path0'], 'memmap')):
            os.makedirs(os.path.join(ops['save_path0'], 'memmap'))
        # masks.
        labels, masks, mean_func, max_func, mean_anat, masks_anat = read_masks(ops)
        # trials.
        neural_trials = read_neural_trials(ops, smooth)
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
        list_neural_trials.append(neural_trials)
        list_move_offset.append([xoff, yoff])
        list_significance.append(significance)
        # clear memory usages.
        del labels
        del masks, mean_func, max_func, mean_anat, masks_anat
        del neural_trials
        del xoff, yoff
        del significance
        gc.collect()
    return [list_labels, list_masks,
            list_neural_trials, list_move_offset, list_significance]

# get session results for all subject.
def read_all(session_config_list, smooth):
    list_labels = []
    list_masks = []
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
        labels, masks, neural_trials, move_offset, significance = read_subject(
             list_ops,
             sig_tag=session_config_list['list_config'][i]['sig_tag'],
             force_label=session_config_list['list_config'][i]['force_label'],
             smooth=smooth)
        # append to list.
        list_labels += labels
        list_masks += masks
        list_neural_trials += neural_trials
        list_move_offset += move_offset
        list_significance += significance
        # clear memory usages.
        del labels
        del masks
        del neural_trials
        del move_offset
        del significance
        gc.collect()
    return [list_labels, list_masks,
            list_neural_trials, list_move_offset, list_significance]
    
# clean memory mapping files.
def clean_memap_path(ops):
    try:
        if os.path.exists(os.path.join(ops['save_path0'], 'memmap')):
            shutil.rmtree(os.path.join(ops['save_path0'], 'memmap'))
    except: pass
    