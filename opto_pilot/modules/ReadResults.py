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
    dff = np.array(f['name'])
    f.close()
    return dff


# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops, cate_delay):
    # read h5 file.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    partition = 4
    # resort delay settings.
    start = np.min(np.array([t for t in neural_trials.keys()]).astype('int32'))
    end   = np.max(np.array([t for t in neural_trials.keys()]).astype('int32'))
    trial_idx = np.arange(start, end+1)
    trial_delay = np.array([neural_trials[str(t)]['trial_delay'] for t in trial_idx])
    # mark short and long delay trials short:0 long:1.
    trial_type = np.zeros_like(trial_delay)
    trial_type[trial_delay>cate_delay] = 1
    # mark epoch trials unvalid:-1 early:1 late:0
    block_change = np.diff(trial_type, prepend=0)
    block_change[block_change!=0] = 1

    block_change[0] = 1
    block_change[-1] = 1
    block_change = np.where(block_change==1)[0]
    block_epoch = np.zeros_like(trial_type)
    for start, end in zip(block_change[:-1], block_change[1:]):
        tran = start + (end - start) // partition
        block_epoch[start:tran] = 1
    block_epoch[:block_change[1]] = -1
    # write into neural trials.
    for i in range(len(trial_idx)):
        neural_trials[str(trial_idx[i])]['trial_type'] = trial_type[i]
        neural_trials[str(trial_idx[i])]['block_epoch'] = block_epoch[i]
    return neural_trials


# read significance test label results.
def read_significance(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'r')
    significance = {}

#     significance['r_vis'] = np.array(f['significance']['r_vis'])
#     significance['r_push'] = np.array(f['significance']['r_push'])
#     significance['r_retract'] = np.array(f['significance']['r_retract'])
#     significance['r_wait'] = np.array(f['significance']['r_wait'])
#     significance['r_reward'] = np.array(f['significance']['r_reward'])
#     significance['r_punish'] = np.array(f['significance']['r_punish'])
#     significance['r_lick'] = np.array(f['significance']['r_lick'])

    significance['r_vis1']     = np.array(f['significance']['r_vis1'])
    significance['r_push1']    = np.array(f['significance']['r_push1'])
    significance['r_retract1'] = np.array(f['significance']['r_retract1'])
    significance['r_vis2']     = np.array(f['significance']['r_vis2'])
    significance['r_push2']    = np.array(f['significance']['r_push2'])
    significance['r_retract2'] = np.array(f['significance']['r_retract2'])
    significance['r_wait']     = np.array(f['significance']['r_wait'])
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
        if ndarray.ndim > 0:
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
              
        
        #print(interpolator)
        new_time = np.arange(0, 60000, 1)
        # interpolator = interp1d(js_time, js_pos, bounds_error=False)
        # new_pos = interpolator(new_time)
        new_pos = np.interp(new_time , js_time, js_pos)
        idx_start = np.argmin(np.abs(new_time - start_time))
        idx_end = np.argmin(np.abs(new_time - end_time))
        # print(new_pos)
        # print(interpolator)
        new_pos = savgol_filter(new_pos, window_length=40, polyorder=3)
        vel = np.gradient(new_pos, new_time)
        vel = savgol_filter(vel, window_length=40, polyorder=1)
        onset4velocity = velocity_onset(vel, idx_start, idx_end)
        if onset4velocity[0] == 0:
            push = np.array([np.nan])
        else:
            push = np.array([new_time[onset4velocity[0]]])
        return push
    
    def states_labeling(trial_states, reps):
        if ('Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0])) or ('EarlyPressPunish' in trial_states.keys() and not np.isnan(trial_states['EarlyPressPunish'][0])) or ('EarlyPress1Punish' in trial_states.keys() and not np.isnan(trial_states['EarlyPress1Punish'][0])) or ('EarlyPress2Punish' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2Punish'][0]))or ('LatePress1' in trial_states.keys() and not np.isnan(trial_states['LatePress1'][0]))or ('LatePress2' in trial_states.keys() and not np.isnan(trial_states['LatePress2'][0])):
            if 'DidNotPress1' in trial_states.keys() and not np.isnan(trial_states['DidNotPress1'][0]):
                outcome = 'DidNotPress1'
            elif 'DidNotPress2' in trial_states.keys() and not np.isnan(trial_states['DidNotPress2'][0]):
                outcome = 'DidNotPress2'
            elif 'DidNotPress3' in trial_states.keys() and not np.isnan(trial_states['DidNotPress3'][0]):
                outcome = 'DidNotPress3'
            elif 'EarlyPress' in trial_states.keys() and not np.isnan(trial_states['EarlyPress'][0]):
                outcome = 'EarlyPress'
            elif 'EarlyPress1' in trial_states.keys() and not np.isnan(trial_states['EarlyPress1'][0]):
                outcome = 'EarlyPress1'
            elif 'EarlyPress2' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2'][0]):
                outcome = 'EarlyPress2'
            elif 'LatePress1' in trial_states.keys() and not np.isnan(trial_states['LatePress1'][0]):
                outcome = 'LatePress1'
            elif 'LatePress2' in trial_states.keys() and not np.isnan(trial_states['LatePress2'][0]):
                outcome = 'LatePress2'
            else:
                outcome = 'Punish'
        elif reps == 1 and 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
            outcome = 'Reward'
        elif reps == 2 and 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
            outcome = 'Reward'
        elif reps == 3 and 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):        
            outcome = 'Reward'
        elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
            outcome = 'Reward'
        elif 'VisStimInterruptDetect1' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptDetect1'][0]):
            outcome = 'VisStimInterruptDetect1'
        elif 'VisStimInterruptDetect2' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptDetect2'][0]):
            outcome = 'VisStimInterruptDetect2'
        elif 'VisStimInterruptGray1' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptGray1'][0]):
            outcome = 'VisStimInterruptGray1'
        elif 'VisStimInterruptGray2' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptGray2'][0]):
            outcome = 'VisStimInterruptGray2'
        else:
            outcome = 'Other' # VisInterrupt
        
        return outcome
    
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
        trial_ST = []
        trial_js_pos = []
        trial_js_time = []
        trial_outcome = []
        trial_retract1_init = []
        for i in range(raw['nTrials']):
            trial_states = raw['RawEvents']['Trial'][i]['States']
            trial_events = raw['RawEvents']['Trial'][i]['Events']
            
            trial_outcome.append(states_labeling(trial_states, 2))
            
            # push1 onset.
            start = 1000*np.array(trial_states['VisDetect1'][0]).reshape(-1)-500
            end = 1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1)-100
            push1 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            
            # push2 onset.
            start = 1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1)
            if ('RotaryEncoder1_1' in trial_events.keys() and np.size(trial_events['RotaryEncoder1_1'])>1):

                end = 1000*np.array(trial_events['RotaryEncoder1_1'][1]).reshape(-1)

            else:
                end = np.array([np.nan])
            push2 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            if np.isnan(push1):
                push2 = np.array([np.nan])
            trial_push1.append(push1)
            trial_push2.append(push2)

            trial_vis1.append(1000*np.array(trial_states['VisualStimulus1']).reshape(-1))
            # 1st retract.
            trial_retract1.append(1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1))
            # 1st retract.
            trial_retract1_init.append(1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1))
            # 2nd stim.
            trial_vis2.append(1000*np.array(trial_states['VisualStimulus2']).reshape(-1))
            # wait for 2nd push.
            trial_wait2.append(1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1))
            # 2nd retract.
            if ('LeverRetractFinal' in trial_states.keys()):
                trial_retract2.append(1000*np.array(trial_states['LeverRetractFinal'][0]).reshape(-1))
            else:
                trial_retract2.append(1000*np.array([np.nan, np.nan]).reshape(-1))
            # reward.
            trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
            # punish.
            if not np.isnan(trial_states['Punish'][0]):
                trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
            else:
                trial_punish.append(1000*np.array(trial_states['EarlyPress2Punish']).reshape(-1))
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
            # self timed / visually guide
            trial_GUI_Params = raw['TrialSettings'][i]['GUI']
            # mode: vis-guided or self timed
            isSelfTimedMode = 0    # assume vis-guided (all sessions were vi)
            if 'SelfTimedMode' in trial_GUI_Params:
                isSelfTimedMode = trial_GUI_Params['SelfTimedMode']
                
            trial_ST.append(isSelfTimedMode)
            # iti.
            trial_iti.append(1000*np.array(trial_states['ITI']).reshape(-1))
            # delay
            
            if np.min(raw['TrialTypes']) == np.max(raw['TrialTypes']):
                #print('single bock session')
                #print(np.min(raw['TrialTypes']), np.max(raw['TrialTypes']))
                #print(len(raw['TrialTypes']))
                #trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
                ## added temprary
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
                ## end adding
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
            # joystick trajectory.
            js_pos = np.array(raw['EncoderData'][i]['Positions'])
            js_time = 1000*np.array(raw['EncoderData'][i]['Times'])
            if np.abs(js_pos[0]) > 0.9 or np.abs(js_time[0]) > 1e-5:
                trial_js_pos.append(np.array([0,0,0,0,0]))
                trial_js_time.append(np.array([0,1,2,3,4]))
            else:
                trial_js_pos.append(js_pos)
                trial_js_time.append(js_time)
                
                
        bpod_sess_data = {

            'trial_types'        : np.array(raw['TrialTypes']),
            'trial_vis1'         : trial_vis1,
            'trial_push1'        : trial_push1,
            'trial_retract1'     : trial_retract1,
            'trial_retract1_init': trial_retract1_init,
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
            'trial_ST'           : trial_ST,
            'trial_js_pos'       : trial_js_pos,
            'trial_js_time'      : trial_js_time,
            'trial_outcome'      : trial_outcome,
            }

        return bpod_sess_data
    bpod_sess_data = main()
    
    return bpod_sess_data
