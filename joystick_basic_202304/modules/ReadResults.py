#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio


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
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials


# read significance test label results.
def read_significance(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'r')
    significance = {}
    significance['r_vis1']     = np.array(f['significance']['r_vis1'])
    significance['r_press1']   = np.array(f['significance']['r_press1'])
    significance['r_retract1'] = np.array(f['significance']['r_retract1'])
    significance['r_vis2']     = np.array(f['significance']['r_vis2'])
    significance['r_press2']   = np.array(f['significance']['r_press2'])
    significance['r_reward']   = np.array(f['significance']['r_reward'])
    significance['r_punish']   = np.array(f['significance']['r_punish'])
    significance['r_lick']     = np.array(f['significance']['r_lick'])
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
    trial_vis1 = []
    trial_press1 = []
    trial_retract1 = []
    trial_vis2 = []
    trial_press2 = []
    trial_reward = []
    trial_punish = []
    trial_lick = []
    trial_js_pos = []
    trial_js_time = []
    for i in range(raw['nTrials']):
        trial_states = raw['RawEvents']['Trial'][i]['States']
        trial_events = raw['RawEvents']['Trial'][i]['Events']
        # 1st stim.
        trial_vis1.append(1000*np.array(trial_states['VisualStimulus1']).reshape(-1))
        # 1st press window.
        trial_press1.append(1000*np.array(trial_states['WaitForPress1'][1]).reshape(-1))
        # 1st retract.
        trial_retract1.append(1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1))
        # 2nd stim.
        trial_vis2.append(1000*np.array(trial_states['VisualStimulus2']).reshape(-1))
        # 2nd press window.
        trial_press2.append(1000*np.array(trial_states['WaitForPress2'][1]).reshape(-1))
        # reward.
        trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
        # punish.
        trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
        # licking events.
        if 'Port2In' in trial_events.keys():
            trial_lick.append(1000*np.array(trial_events['Port2In']).reshape(-1))
        else:
            trial_lick.append(np.array([np.nan]))
        # joystick position.
        trial_js_pos.append(np.array(raw['EncoderData'][i]['Positions']).reshape(-1))
        # joystick timestamps.
        trial_js_time.append(1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1))
    bpod_sess_data = {
        'trial_types'    : np.array(raw['TrialTypes']),
        'trial_vis1'     : trial_vis1,
        'trial_press1'   : trial_press1,
        'trial_retract1' : trial_retract1,
        'trial_vis2'     : trial_vis2,
        'trial_press2'   : trial_press2,
        'trial_reward'   : trial_reward,
        'trial_punish'   : trial_punish,
        'trial_lick'     : trial_lick,
        'trial_js_pos'   : trial_js_pos,
        'trial_js_time'  : trial_js_time,
        }
    return bpod_sess_data

