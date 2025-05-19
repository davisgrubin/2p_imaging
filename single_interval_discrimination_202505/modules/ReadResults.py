#!/usr/bin/env python3

import gc
import os
import copy
import h5py
import shutil
import numpy as np
import pandas as pd
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
def read_camera(ops):
    mm_path, file_path = get_memmap_path(ops, 'camera.h5')
    try:
        with h5py.File(file_path, 'r') as f:
            camera_time = create_memmap(f['camera']['camera_time'], 'float32', os.path.join(mm_path, 'camera_time.mmap'))
            pupil       = create_memmap(f['camera']['pupil'],       'float32', os.path.join(mm_path, 'pupil.mmap'))
    except:
        camera_time = create_memmap(np.array([np.nan]), 'float32', os.path.join(mm_path, 'camera_time.mmap'))
        pupil       = create_memmap(np.array([np.nan]),       'float32', os.path.join(mm_path, 'pupil.mmap'))
    return [camera_time, pupil]

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

# read trial label csv file into dataframe.
def read_trial_label(ops):
    raw_csv = pd.read_csv(os.path.join(ops['save_path0'], 'trial_labels.csv'), index_col=0)
    # recover object numpy array from csv str.
    def object_parse(k, shape):
        arr = np.array(
            [np.fromstring(s.replace('[', '').replace(']', ''), sep=' ').reshape(shape)
             for s in raw_csv[k].to_list()] + ['yicong_forever'],
            dtype='object')[:-1]
        return arr
    # parse all array.
    time_trial_start = raw_csv['time_trial_start'].to_numpy(dtype='float32')
    time_trial_end = raw_csv['time_trial_end'].to_numpy(dtype='float32')
    trial_type = raw_csv['trial_type'].to_numpy(dtype='int8')
    outcome = raw_csv['outcome'].to_numpy(dtype='object')
    state_window_choice = object_parse('state_window_choice', [-1])
    state_reward = object_parse('state_reward', [-1])
    state_punish = object_parse('state_punish', [-1])
    stim_seq = object_parse('stim_seq', [-1,2])
    isi = raw_csv['isi'].to_numpy(dtype='float32')
    lick = object_parse('lick', [4,-1])
    # convert to dataframe.
    trial_labels = pd.DataFrame({
        'time_trial_start': time_trial_start,
        'time_trial_end': time_trial_end,
        'trial_type': trial_type,
        'outcome': outcome,
        'state_window_choice': state_window_choice,
        'state_reward': state_reward,
        'state_punish': state_punish,
        'stim_seq': stim_seq,
        'isi': isi,
        'lick': lick,
        })
    return trial_labels

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops, smooth):
    mm_path, file_path = get_memmap_path(ops, 'neural_trials.h5')
    trial_labels = read_trial_label(ops)
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
        neural_trials['trial_labels'] = trial_labels
        neural_trials['vol_time']     = create_memmap(f['neural_trials']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        neural_trials['vol_stim_vis'] = create_memmap(f['neural_trials']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        neural_trials['vol_stim_aud'] = create_memmap(f['neural_trials']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        neural_trials['vol_flir']     = create_memmap(f['neural_trials']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        neural_trials['vol_pmt']      = create_memmap(f['neural_trials']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        neural_trials['vol_led']      = create_memmap(f['neural_trials']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return neural_trials

# read significance test label results.
def read_significance(ops):
    file_path = os.path.join(ops['save_path0'], 'significance.h5')
    with h5py.File(file_path, 'r') as f:
        significance = {}
        significance['r_all'] = np.array(f['significance']['r_all'])
    return significance

# read bpod session data.
def read_bpod_mat_data(ops, session_start_time):
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
    # labeling every trials for a subject
    def states_labeling(trial_states):
        if 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
            outcome = 'punish'
        elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
            outcome = 'reward'
        elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
            outcome = 'naive_punish'
        elif 'RewardNaive' in trial_states.keys() and not np.isnan(trial_states['RewardNaive'][0]):
            outcome = 'naive_reward'
        elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
            outcome = 'no_choose'
        else:
            outcome = 'other'
        return outcome
    # find state timing.
    def get_state(trial_state_dict, target_state, trial_start):
        if target_state in trial_state_dict:
            time_state = 1000*np.array(trial_state_dict[target_state]) + trial_start
        else:
            time_state = np.array([np.nan, np.nan])
        return time_state
    # read raw data.
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']
    trial_labels = dict()
    n_trials = raw['nTrials']
    trial_states = [raw['RawEvents']['Trial'][ti]['States'] for ti in range(n_trials)]
    trial_events = [raw['RawEvents']['Trial'][ti]['Events'] for ti in range(n_trials)]
    # trial start time stamps.
    trial_labels['time_trial_start'] = 1000*np.array(raw['TrialStartTimestamp']).reshape(-1)
    # trial end time stamps.
    trial_labels['time_trial_end'] = 1000*np.array(raw['TrialEndTimestamp']).reshape(-1)
    # correct timestamps starting from session start.
    trial_labels['time_trial_end'] = trial_labels['time_trial_end'] - trial_labels['time_trial_start'][0] + session_start_time
    trial_labels['time_trial_start'] = trial_labels['time_trial_start'] - trial_labels['time_trial_start'][0] + session_start_time
    # trial target.
    trial_labels['trial_type'] = np.array(raw['TrialTypes']).reshape(-1)-1
    # trial outcomes.
    trial_labels['outcome'] = np.array([states_labeling(ts) for ts in trial_states], dtype='object')
    # trial state timings.
    trial_labels['state_window_choice'] = np.array([
        get_state(trial_states[ti], 'WindowChoice', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_reward'] = np.array([
        get_state(trial_states[ti], 'Reward', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_punish'] = np.array([
        get_state(trial_states[ti], 'Punish', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # stimulus timing.
    trial_isi = []
    trial_stim_seq = []
    for ti in range(n_trials):
        # stimulus sequence.
        if ('BNC1High' in trial_events[ti].keys() and
            'BNC1Low' in trial_events[ti].keys() and
            len(np.array(trial_events[ti]['BNC1High']).reshape(-1))==2 and
            len(np.array(trial_events[ti]['BNC1Low']).reshape(-1))==2
            ):
            stim_seq = 1000*np.array([trial_events[ti]['BNC1High'], trial_events[ti]['BNC1Low']]) + trial_labels['time_trial_start'][ti]
            stim_seq = np.transpose(stim_seq, [1,0])
            isi = 1000*np.array(trial_events[ti]['BNC1High'][1] - trial_events[ti]['BNC1Low'][0])
        else:
            stim_seq = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            isi = np.nan
        trial_stim_seq.append(stim_seq)
        trial_isi.append(isi)
    trial_labels['stim_seq'] = np.array(trial_stim_seq + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['isi'] = np.array(trial_isi + ['yicong_forever'], dtype='object')[:-1]
    # licking.
    trial_lick = []
    for ti in range(n_trials):
        licking_events = []
        # 0 left 1 right.
        direction = []
        # 0 wrong 1 correct.
        correctness = []
        # left.
        if 'Port1In' in trial_events[ti].keys():
            lick_left = np.array(trial_events[ti]['Port1In']).reshape(-1)
            licking_events.append(lick_left)
            direction.append(np.zeros_like(lick_left))
            if trial_labels['trial_type'][ti] == 0:
                correctness.append(np.ones_like(lick_left))
            else:
                correctness.append(np.zeros_like(lick_left))
        # right.
        if 'Port3In' in trial_events[ti].keys():
            lick_right = np.array(trial_events[ti]['Port3In']).reshape(-1)
            licking_events.append(lick_right)
            direction.append(np.ones_like(lick_right))
            if trial_labels['trial_type'][ti] == 1:
                correctness.append(np.ones_like(lick_right))
            else:
                correctness.append(np.zeros_like(lick_right))
        if len(licking_events) > 0:
            # combine all licking.
            licking_events = 1000*np.concatenate(licking_events).reshape(1,-1) + trial_labels['time_trial_start'][ti]
            direction = np.concatenate(direction).reshape(1,-1)
            correctness = np.concatenate(correctness).reshape(1,-1)
            lick = np.concatenate([licking_events, direction, correctness], axis=0)
            # sort based on timing.
            lick = lick[:,np.argsort(lick[0,:])]
            # filter false detection before licking window.
            lick = lick[:,lick[0,:] >= trial_labels['state_window_choice'][ti][0]]
            # classify licking.
            if np.size(lick) != 0:
                lick_type = np.full(lick.shape[1], np.nan)
                lick_type[0] = 1
                if (not np.isnan(trial_labels['state_reward'][ti][1]) and
                    len(lick_type) > 1
                    ):
                    lick_type[1:][lick[0,1:] > trial_labels['state_reward'][ti][0]] = 0
                lick_type = lick_type.reshape(1,-1)
                lick = np.concatenate([lick, lick_type], axis=0)
            else:
                lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        else:
            lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        # all licking events.
        trial_lick.append(lick)
    trial_labels['lick'] = np.array(trial_lick + ['yicong_forever'], dtype='object')[:-1]
    # convert to dataframe.
    trial_labels = pd.DataFrame(trial_labels)
    return trial_labels

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
            significance['r_all'] = np.ones_like(significance['r_all']).astype('bool')
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
    