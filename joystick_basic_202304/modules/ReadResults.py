#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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
    significance['r_vis']     = np.array(f['significance']['r_vis'])
    significance['r_push']    = np.array(f['significance']['r_push'])
    significance['r_retract'] = np.array(f['significance']['r_retract'])
    significance['r_wait']    = np.array(f['significance']['r_wait'])
    significance['r_reward']  = np.array(f['significance']['r_reward'])
    significance['r_punish']  = np.array(f['significance']['r_punish'])
    significance['r_lick']    = np.array(f['significance']['r_lick'])
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
    def get_push_onset(js_pos, js_time, start_time, end_time):
        def find_half_peak_point_before(velocity, peak_idx):
            half_peak_value = 2.5
            for i in range(peak_idx - 1, -1, -1):
                if velocity[i] <= half_peak_value:
                    return i
            return 0
        def find_onethird_peak_point_before(velocity, peak_idx):
            peak_value = velocity[peak_idx]
            onethird_peak_value = peak_value * 0.4
            for i in range(peak_idx - 1, -1, -1):
                if velocity[i] <= onethird_peak_value:
                    return i
            return 0
        def velocity_onset(js_pos, start, end):
            start = max(0,start)
            end = min(len(js_pos), end)
            peaks,_ = find_peaks(js_pos[start:end],distance=65, height=5)
            onset4velocity = []
            if len(peaks) >= 1:
                peaks = peaks + start
            if len(peaks) == 0:
                peaks = end
                onset4velocity.append(find_onethird_peak_point_before(js_pos,peaks))
                return onset4velocity
            if len(peaks) >= 1:
                peaks = np.hstack((peaks,end))
                for i in range(0, len(peaks)):
                    onset4velocity.append(find_onethird_peak_point_before(js_pos, peaks[i]))
                return onset4velocity
        interpolator = interp1d(js_time, js_pos, bounds_error=False)
        new_time = np.arange(np.min(js_time), np.max(js_time), 1)
        new_pos = interpolator(new_time)
        idx_start = np.argmin(np.abs(new_time - start_time))
        idx_end = np.argmin(np.abs(new_time - end_time))
        new_pos = savgol_filter(new_pos, window_length=40, polyorder=3)
        vel = np.gradient(new_pos, new_time)
        vel = savgol_filter(vel, window_length=40, polyorder=1)
        onset4velocity = velocity_onset(vel, idx_start, idx_end)
        if onset4velocity[0] == 0:
            push = np.array([np.nan])
        else:
            push = np.array([new_time[onset4velocity[0]]])
        return push
    def main():
        raw = sio.loadmat(
            os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
            struct_as_record=False, squeeze_me=True)
        raw = _check_keys(raw)['SessionData']
        trial_vis1 = []
        trial_push1 = []
        trial_retract1 = []
        trial_vis2 = []
        trial_wait2 = []
        trial_push2 = []
        trial_retract2 = []
        trial_reward = []
        trial_punish = []
        trial_no1stpush = []
        trial_no2ndpush = []
        trial_early2ndpush = []
        trial_iti = []
        trial_lick = []
        trial_delay = []
        trial_js_pos = []
        trial_js_time = []
        for i in range(raw['nTrials']):
            trial_states = raw['RawEvents']['Trial'][i]['States']
            trial_events = raw['RawEvents']['Trial'][i]['Events']
            # push onset.
            push1 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                1000*np.array(trial_states['VisDetect1'][0]).reshape(-1),
                1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1))
            push2 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1),
                1000*np.array(trial_states['ITI'][0]).reshape(-1))
            if np.isnan(push1):
                push2 = np.array([np.nan])
            trial_push1.append(push1)
            trial_push2.append(push2)
            # 1st stim.
            trial_vis1.append(1000*np.array(trial_states['VisualStimulus1']).reshape(-1))
            # 1st retract.
            trial_retract1.append(1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1))
            # 2nd stim.
            trial_vis2.append(1000*np.array(trial_states['VisualStimulus2']).reshape(-1))
            # wait for 2nd push.
            trial_wait2.append(1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1))
            # 2nd retract.
            trial_retract2.append(1000*np.array(trial_states['LeverRetract2'][0]).reshape(-1))
            # reward.
            trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
            # punish.
            trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
            # did not push 1.
            trial_no1stpush.append(1000*np.array(trial_states['DidNotPress1']).reshape(-1))
            # did not push 2.
            trial_no2ndpush.append(1000*np.array(trial_states['DidNotPress2']).reshape(-1))
            # early push 2.
            trial_early2ndpush.append(1000*np.array(trial_states['EarlyPress2']).reshape(-1))
            # licking events.
            if 'Port2In' in trial_events.keys():
                lick_all = 1000*np.array(trial_events['Port2In']).reshape(1,-1)
                lick_label = np.zeros_like(lick_all).reshape(1,-1)
                lick_label[lick_all>1000*np.array(trial_states['Reward'][0])] = 1
                trial_lick.append(np.concatenate((lick_all, lick_label), axis=0))
            else:
                trial_lick.append(np.array([[np.nan],[np.nan]]))
            # iti.
            trial_iti.append(1000*np.array(trial_states['ITI']).reshape(-1))
            # delay
            if np.min(raw['TrialTypes']) == np.max(raw['TrialTypes']):
                trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
            else:
                if 'PrePress2DelayShort_s' in raw['TrialSettings'][i]['GUI'].keys():
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayLong_s'])
                else:
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
            # joystick position.
            trial_js_pos.append(np.array(raw['EncoderData'][i]['Positions']).reshape(-1))
            # joystick timestamps.
            trial_js_time.append(1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1))
        bpod_sess_data = {
            'trial_types'        : np.array(raw['TrialTypes']),
            'trial_vis1'         : trial_vis1,
            'trial_push1'        : trial_push1,
            'trial_retract1'     : trial_retract1,
            'trial_vis2'         : trial_vis2,
            'trial_wait2'        : trial_wait2,
            'trial_push2'        : trial_push2,
            'trial_retract2'     : trial_retract2,
            'trial_reward'       : trial_reward,
            'trial_punish'       : trial_punish,
            'trial_no1stpush'    : trial_no1stpush,
            'trial_no2ndpush'    : trial_no2ndpush,
            'trial_early2ndpush' : trial_early2ndpush,
            'trial_iti'          : trial_iti,
            'trial_lick'         : trial_lick,
            'trial_delay'        : trial_delay,
            'trial_js_pos'       : trial_js_pos,
            'trial_js_time'      : trial_js_time,
            }
        return bpod_sess_data
    bpod_sess_data = main()
    return bpod_sess_data
