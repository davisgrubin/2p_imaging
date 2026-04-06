#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd

from Modules.ReadResults import read_raw_voltages
from Modules.ReadResults import read_dff
from Modules.ReadResults import read_spikes
from Modules.ReadResults import read_bpod_mat_data
from Modules.ReadResults import read_camera
from scipy.interpolate import interp1d


# detect the rising edge and falling edge of binary series.
def find_cor(df, bodypart = "eyeLeft"):
    x = df.loc[:, (slice(None), bodypart, "x")].iloc[:, 0]
    y = df.loc[:, (slice(None), bodypart, "y")].iloc[:, 0]
    likelihood = df.loc[:, (slice(None), bodypart, "likelihood")].iloc[:, 0]

    return x.copy(), y.copy(), likelihood.copy()

def cleean_data(x, y, likelihood, thereshold = 0.7):
    x[likelihood < thereshold] = np.nan
    y[likelihood < thereshold] = np.nan

    return x, y

def polygon_area(x_coords, y_coords):
    
    return 0.5 * np.abs(
        np.dot(x_coords, np.roll(y_coords, -1)) -
        np.dot(y_coords, np.roll(x_coords, -1))
    )


def dlc_to_h5(ops, fps=30, likelihood_threshold=0.7):

    csv_path = os.path.join(ops['save_path0'], "dlc_output.csv")
    h5_path = os.path.join(ops['save_path0'], "camera_data.h5")

    df = pd.read_csv(csv_path, header=[0,1,2])

    # bodyparts used to define pupil boundary
    bodyparts = ['pupilTop', 'pupilRight', 'pupilBottom', 'pupilLeft']

    n_frames = len(df)
    pupil = np.zeros(n_frames)

    # extract coordinates for each bodypart
    coords = {}
    for part in bodyparts:
        x, y, likelihood = find_cor(df, bodypart=part)
        x, y = cleean_data(x, y, likelihood, thereshold=likelihood_threshold)
        coords[part] = (x, y)

    # compute polygon area frame-by-frame
    for i in range(n_frames):

        x_frame = []
        y_frame = []

        for part in bodyparts:
            x_frame.append(coords[part][0][i])
            y_frame.append(coords[part][1][i])

        x_frame = np.array(x_frame)
        y_frame = np.array(y_frame)

        if np.isnan(x_frame).any() or np.isnan(y_frame).any():
            pupil[i] = np.nan
        else:
            pupil[i] = polygon_area(x_frame, y_frame)

    pupil = pupil.astype('float32')

    # generate camera time
    camera_time = (np.arange(n_frames) / fps).astype('float32')

    # save H5
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("camera_dlc")
        grp.create_dataset("camera_time", data=camera_time)
        grp.create_dataset("pupil", data=pupil)

    print("Saved:", h5_path)
def covert_to_arr(times, ref):
    output = np.zeros_like(ref)
    for t in times:
        indx = np.argmin(np.abs(ref-t))
        output[indx] += 1 
    return output

def add_event_aray(neural_data, neural_trials, event_list, info_list):
    all_events_array = []
    for event in event_list:
        all_events_array.append(np.zeros_like(neural_data['time']))
        
    all_info_array = []
    for info in info_list:
        all_info_array.append(np.zeros_like(neural_data['time']))
        
    time = neural_data['time']
    js_time = []
    lick_time = []
    js_pos = []
    interval = 1
    trial_start = np.where(neural_data['trial_start'] == 1)[0]
    start_time = time[trial_start]
    for trial in neural_trials.keys():
        js_time_trial = neural_trials[trial]['trial_js_time']
        js_pos_trial = neural_trials[trial]['trial_js_pos']
        interpolator = interp1d(js_time_trial, js_pos_trial, bounds_error=False) 
        new_time = np.arange(np.min(js_time_trial), np.max(js_time_trial), interval) 
        new_pos = interpolator(new_time) 
        js_time.append(new_time) 
        js_pos.append(new_pos)
        lick_time.append(neural_trials[trial]['trial_all_lick'])
        
        for i, event in enumerate(event_list):
            event_time = neural_trials[trial][event]
            if not np.isnan(event_time[0]):
                indx_start = np.argmin(np.abs(time-event_time[0]))
                all_events_array[i][indx_start] = 1
                indx_end = np.argmin(np.abs(time-event_time[-1]))
                if indx_start == 0:
                    print(event, trial, event_time[0])
                if not indx_start == indx_end:
                    all_events_array[i][indx_end] = -1
                
        first_time = neural_trials[trial]['vol_time'][0]
        idx_iniiation = np.searchsorted(start_time, first_time) - 1
        idx_iniiation = np.argmin(np.abs(start_time-first_time))
        if idx_iniiation >= 0:
            start_trial = trial_start[idx_iniiation]
        else:
            start_trial = -1
            
        for i, info in enumerate(info_list):
            all_info_array[i][start_trial] = neural_trials[trial][info]
                
    for i, event in enumerate(event_list):
        neural_data[event[6:]] = all_events_array[i]
        
    for i, info in enumerate(info_list):
        neural_data[info[6:]] = all_info_array[i]
        
        
    neural_data['js_time'] = np.concatenate(js_time)
    neural_data['js_pos'] = np.concatenate(js_pos)
    neural_data['lick_time'] = np.concatenate(lick_time)
    neural_data['lick'] = covert_to_arr(np.concatenate(lick_time), time)
    
    return neural_data
    

def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down


# correct the fluorescence signal timing.

def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro


# align the stimulus sequence with fluorescence signal.
def align_stim(
        vol_time,
        time_neuro,
        vol_stim_vis,
        label_stim,
        ):
    # find the rising and falling time of stimulus.
    stim_time_up, stim_time_down = get_trigger_time(
        vol_time, vol_stim_vis)
    # avoid going up but not down again at the end.
    stim_time_up = stim_time_up[:len(stim_time_down)]
    # assign the start and end time to fluorescence frames.
    stim_start = []
    stim_end = []
    for i in range(len(stim_time_up)):
        # find the nearest frame that stimulus start or end.
        stim_start.append(
            np.argmin(np.abs(time_neuro - stim_time_up[i])))
        stim_end.append(
            np.argmin(np.abs(time_neuro - stim_time_down[i])))
    # reconstruct stimulus sequence.
    stim = np.zeros(len(time_neuro))
    for i in range(len(stim_start)):
        label = label_stim[vol_time==stim_time_up[i]][0]
        stim[stim_start[i]:stim_end[i]] = label
    return stim


# process trial start signal.

def get_trial_start_end(
        vol_time,
        vol_start,
        ):
    time_up, time_down = get_trigger_time(vol_time, vol_start)
    # find the impulse start signal.
    time_start = [time_up[0]]
    for i in range(len(time_up)-1):
        if time_up[i+1] - time_up[i] > 5:
            time_start.append(time_up[i+1])
    start = []
    end = []
    # assume the current trial end at the next start point.
    for i in range(len(time_start)):
        s = time_start[i]
        e = time_start[i+1] if i != len(time_start)-1 else -1
        start.append(s)
        end.append(e)
    return start, end


# trial segmentation.
def trial_split(
        start, end,
        dff,spikes, 
        stim, time_neuro,
        time_pupil,
        label_stim, vol_time,
        ):
    neural_trials = dict()
    for i in range(len(start)):
        if np.max(time_neuro > start[i]):
            neural_trials[str(i)] = dict()
            #print(i)
            start_idx_dff = np.where(time_neuro > start[i])[0][0]
            end_idx_dff   = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
            neural_trials[str(i)]['time'] = time_neuro[start_idx_dff:end_idx_dff]
            neural_trials[str(i)]['stim'] = stim[start_idx_dff:end_idx_dff]
            start_idx_pupil = np.where(time_pupil > start[i])[0][0]
            end_idx_pupil   = np.where(time_pupil < end[i])[0][-1] if end[i] != -1 else -1
            neural_trials[str(i)]['pupil_time'] = time_pupil[start_idx_pupil:end_idx_pupil]
            neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
            neural_trials[str(i)]['spikes'] = spikes[:,start_idx_dff:end_idx_dff]
            start_idx_vol = np.where(vol_time > start[i])[0][0]
            end_idx_vol   = np.where(vol_time < end[i])[0][-1] if end[i] != -1 else -1
            neural_trials[str(i)]['vol_stim'] = label_stim[start_idx_vol:end_idx_vol]
            neural_trials[str(i)]['vol_time'] = vol_time[start_idx_vol:end_idx_vol]
    return neural_trials


# add trial information with bpod session data.
def trial_label(
        ops,
        neural_trials,
        ):
    bpod_sess_data = read_bpod_mat_data(ops)
    #print(len(bpod_sess_data['trial_types']))
    #print(len(neural_trials))
    for i in range(np.min([len(neural_trials), len(bpod_sess_data['trial_types'])])):
        if i-1 in range(np.min([len(neural_trials), len(bpod_sess_data['trial_types'])])):
            neural_trials[str(i)]['pre_trial'] = i-1
            #print(neural_trials[str(i)]['pre_trial'])
        if i+1 in range(np.min([len(neural_trials), len(bpod_sess_data['trial_types'])])):
            neural_trials[str(i)]['post_trial'] = i+1
        neural_trials[str(i)]['trial_types'] = bpod_sess_data[
            'trial_types'][i]
        neural_trials[str(i)]['trial_ST'] = bpod_sess_data[
            'trial_ST'][i]
        neural_trials[str(i)]['trial_delay'] = bpod_sess_data[
            'trial_delay'][i]
        neural_trials[str(i)]['trial_vis1'] = bpod_sess_data[
            'trial_vis1'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_push1'] = bpod_sess_data[
            'trial_push1'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_retract1'] = bpod_sess_data[
            'trial_retract1'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_retract1_init'] = bpod_sess_data[
            'trial_retract1_init'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_vis2'] = bpod_sess_data[
            'trial_vis2'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_wait2'] = bpod_sess_data[
            'trial_wait2'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_push2'] = bpod_sess_data[
            'trial_push2'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_retract2'] = bpod_sess_data[
            'trial_retract2'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_reward'] = bpod_sess_data[
            'trial_reward'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_reward_delay'] = bpod_sess_data[
            'trial_reward_delay'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_punish'] = bpod_sess_data[
            'trial_punish'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_no1stpush'] = bpod_sess_data[
            'trial_no1stpush'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_no2ndpush'] = bpod_sess_data[
            'trial_no2ndpush'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_early2ndpush'] = bpod_sess_data[
            'trial_early2ndpush'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_late2ndpush'] = bpod_sess_data[
            'trial_late2ndpush'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_iti'] = bpod_sess_data[
            'trial_iti'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_lick'] = bpod_sess_data[
            'trial_lick'][i]
        neural_trials[str(i)]['trial_press_delay'] = bpod_sess_data[
            'trial_push2'][i]-bpod_sess_data['trial_retract1'][i][1]
        neural_trials[str(i)]['trial_lick'] += neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_all_lick'] = bpod_sess_data[
            'trial_all_lick'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_js_pos'] = bpod_sess_data[
            'trial_js_pos'][i]
        neural_trials[str(i)]['trial_js_time'] = bpod_sess_data[
            'trial_js_time'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_block'] = bpod_sess_data[
            'trial_block'][i]
        neural_trials[str(i)]['trial_block_len'] = bpod_sess_data[
            'trial_block_len'][i]
        neural_trials[str(i)]['trial_pos_in_block'] = bpod_sess_data[
            'trial_pos_in_block'][i]
        neural_trials[str(i)]['trial_probe'] = bpod_sess_data[
            'trial_probe'][i]
    #print(len(neural_trials))
    return neural_trials

def get_labels(neural_trials, first_epoch = 6, last_epoch = 10):
    """
    takes neural_trials created in trialization; and the number of trials consider first and last epoch.
    will add:
        outcome
        epoch number
        post trial outcome
        inter push delay
        the change of delay with next trial
    to each trial and return the complete version of neural_trials
    """
    print('creating labels')
    all_dt = []
    for trials in neural_trials.keys():
        # epoch label
        if 'trial_pos_in_block' in neural_trials[trials].keys():
            if neural_trials[trials]['trial_pos_in_block'] < first_epoch:
                neural_trials[trials]['block_epoch'] = 0
            elif neural_trials[trials]['trial_block_len'] - neural_trials[trials]['trial_pos_in_block'] < last_epoch:
                neural_trials[trials]['block_epoch'] = 2
            else:
                neural_trials[trials]['block_epoch'] = 1
        else: 
            neural_trials[trials]['block_epoch'] = 1
        # outcome label 
        neural_trials[trials]['trial_outcome'] = get_trial_outcome(neural_trials, trials)
        # post trial outcome label
        if 'post_trial' in neural_trials[trials].keys():
            new_trial = str(neural_trials[trials]['post_trial'])
            if new_trial in neural_trials.keys():
                neural_trials[trials]['post_trial_outcome'] = get_trial_outcome(neural_trials, new_trial)
            else:
                neural_trials[trials]['post_trial_outcome'] = np.nan
        else:
             neural_trials[trials]['post_trial_outcome'] = np.nan
        # the delay deference of current trial and next trial
        neural_trials[trials]['trial_delay_delta'] = get_trial_delta_delay(neural_trials, trials) 
        if not np.isnan(neural_trials[trials]['trial_delay_delta']):
            all_dt.append(neural_trials[trials]['trial_delay_delta'])
    decrease_lim = np.percentile(np.array(all_dt), 30)
    increase_lim = np.percentile(np.array(all_dt), 70)
    #print(all_dt)
    for trials in neural_trials.keys():
        if np.isnan(neural_trials[trials]['trial_delay_delta']):
            neural_trials[trials]['delay_change_label'] = -1
        else:
            if neural_trials[trials]['trial_delay_delta'] < decrease_lim and neural_trials[trials]['trial_delay_delta'] < 0:
                neural_trials[trials]['delay_change_label'] = 1
                print(neural_trials[trials]['trial_delay_delta'])
            elif neural_trials[trials]['trial_delay_delta'] > increase_lim and neural_trials[trials]['trial_delay_delta'] > 0:
                neural_trials[trials]['delay_change_label'] = 2
                print(neural_trials[trials]['trial_delay_delta'])
            else:
                neural_trials[trials]['delay_change_label'] = 0
        
    return neural_trials

        
def get_trial_outcome(neural_trials, trials):
    """
    takes neural_trials andthe trial string.
    returns outcome label for that trial:
        Reward = 0
        DidNotPress1 = 1
        DidNotPress2 = 2
        LatePress2 = 3
        EarlyPress2 = 4
        probe = 5
        other outcomes = -1
    """
    if neural_trials[trials]['trial_probe'] == 1:
        trial_outcome = 5
    elif not np.isnan(neural_trials[trials]['trial_reward'][0]):
        trial_outcome = 0
    elif not np.isnan(neural_trials[trials]['trial_no1stpush'][0]):
        trial_outcome = 1
    elif not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])and (np.isnan(neural_trials[trials]['trial_push2'][0])):
        trial_outcome = 2
    elif (not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])) and (not np.isnan(neural_trials[trials]['trial_push2'][0])):
        #print('late_press1')
        trial_outcome = 3
    elif (not np.isnan(neural_trials[trials]['trial_late2ndpush'][0])):
        #print('late_press2')
        trial_outcome = 3
    elif not np.isnan(neural_trials[trials]['trial_early2ndpush'][0]):
        trial_outcome = 4
    else:
        trial_outcome = -2
    trial_outcome += 1
    return trial_outcome


def get_trial_delta_delay(neural_trials, trials):
    trial_outcome = np.nan
    if 'post_trial' in neural_trials[trials].keys():
        new_trial = str(neural_trials[trials]['post_trial'])
        if new_trial in neural_trials.keys():
            if not np.isnan(neural_trials[trials]['trial_press_delay'][0]):
                if not np.isnan(neural_trials[new_trial]['trial_press_delay'][0]):
                    if neural_trials[trials]['trial_press_delay'][0] > 0 and neural_trials[new_trial]['trial_press_delay'][0] > 0:
                        change_delay = neural_trials[new_trial]['trial_press_delay'][0] - neural_trials[trials]['trial_press_delay'][0]
                        if change_delay < -80:
                            trial_outcome = -1
                        elif change_delay > 80:
                            trial_outcome = 1
                        else: 
                            trial_outcome = 0
    return trial_outcome

def create_neural_data(time_neuro, dff, 
                       spikes,
                       flir_time,
                       camera_time, camera_pupil,
                       start, end, stim):
    neural_data = dict()
    neural_data['time']         = time_neuro
    neural_data['dff']          = dff
    neural_data['spikes']          = spikes
    neural_data['flir_time']          = flir_time
    neural_data['camera_time']  = camera_time
    neural_data['camera_pupil'] = camera_pupil
    neural_data['vis_stim'] = stim
    neural_data['trial_start']  = covert_to_arr(start, time_neuro)
    neural_data['trial_end'] = covert_to_arr(end, time_neuro)
    return neural_data
       
def save_trials(
        ops,
        neural_trials,
        exclude = [5, 5]):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # -- trial_id
    # ---- 1
    # ------ time
    # ------ stim
    # ------ dff
    # ---- 2
    # ...
    if not os.path.exists(os.path.join(ops['save_path0'], 'session_data')):
        os.makedirs(os.path.join(ops['save_path0'], 'session_data'))
        
    f = h5py.File(
        os.path.join(ops['save_path0'], 'session_data', 'neural_trials.h5'),
        'w')
    if len(exclude) == 1: 
        grp = f.create_group('trial_id')
        for trial in neural_trials.keys():
            trial_group = grp.create_group(trial)
            for k in neural_trials[trial].keys():
                trial_group[k] = neural_trials[trial][k]
    else:
        exclude_start = exclude[0]
        exclude_end = exclude[1]
        grp = f.create_group('trial_id')
        for trial in range(len(neural_trials)):
            if trial > exclude_start and trial < len(neural_trials)-exclude_end:
                trial_group = grp.create_group(str(trial))
                for k in neural_trials[str(trial)].keys():
                    trial_group[k] = neural_trials[str(trial)][k]
    f.close()

def save_neural_data(ops, data_dict):
    if not os.path.exists(os.path.join(ops['save_path0'], 'session_data')):
        os.makedirs(os.path.join(ops['save_path0'], 'session_data'))
        
    h5_path = os.path.join(ops['save_path0'], 'session_data', 'neural_data.h5')

    if os.path.exists(h5_path):
        os.remove(h5_path)

    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group('neural_data')

        for key, value in data_dict.items():
            grp[key] = value

# main function for trialization
def run(ops):
    print('===============================================')
    print('=============== trial alignment ===============')
    print('===============================================')
    # --- NEW: Check for DLC and convert if present ---
    dlc_csv_path = os.path.join(ops['save_path0'], "dlc_output.csv")
    if os.path.exists(dlc_csv_path):
        dlc_to_h5(ops)
    else:
        print("No dlc_output.csv found, skipping DLC conversion.")
    # -------------------------------------------------
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops, manual = True)
    print('shape dff', dff.shape)
    dff_z_scored = (dff - dff.mean(axis=1, keepdims=True)) / dff.std(axis=1, keepdims=True)
    spikes = read_spikes(ops, manual = True)
    #print('spike shape',spikes.shape)
    #print('dff shape',dff.shape)
    [vol_time, #time index in ms
     vol_start, #Bpod BNC1 trial start signal from bpod (AI0)
     vol_stim_vis, #sync patch and photodiode (visual stimulus) (AI1)
     vol_img, #ETL scope imaging output (2p microscope image trigger signal) (AI3)
     vol_hifi, #HIFI BNC output. (AI2)
     vol_stim_aud, #Hifi audio output waveform (HIFI waveform signal) (AI4)
     vol_flir,#Flir output (AI5)
     vol_pmt, #PMT Shutter (AI6)
     vol_led, #LED (AI7)
     #vol_2p_stim, #2p stimulation (AI8)
     ] = read_raw_voltages(ops)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps. (gives the start of each 2p frame's time in ms)
    time_img, _   = get_trigger_time(vol_time, vol_img) 
    print('shape time_img', time_img.shape)
    # correct imaging timing. (gives the middle of each 2p frame's time in ms the len will be the number of 2p frames)
    time_neuro = correct_time_img_center(time_img)
    print('shape time_neuro', time_neuro.shape)
    # signal trigger time stamps. (gives the start of each pupil frame's time in ms)
    time_flir, _   = get_trigger_time(vol_time, vol_flir) 
    # correct imaging timing. (gives the middle of each pupil frame's time in ms the len will be the number of pupil frames)
    time_pupil = correct_time_img_center(time_flir)
    # stimulus alignment. (gives an aray with len of 2p frames and in each if the stim was on we have 1)
    print('Aligning stimulus to 2p frame')
    stim = align_stim(vol_time, time_neuro, vol_stim_vis, vol_stim_vis)
    # trial segmentation.
    print('Segmenting trials')
    # start of each trail and end of it in ms (the end of each trial is the start of the next trial. (the last end is -1))
    start, end = get_trial_start_end(vol_time, vol_start)
    # the output dict hase a key for each trial and in each there  are 4 arrays with len of number of 2p frames in that trail and 
    # 2 arrays with the len of number of voltage recording points in that trials (they have 'vol_' in their names)
    neural_trials = trial_split(
        start, end,
        dff_z_scored,spikes, 
        stim, time_neuro,
        time_pupil,
        vol_stim_vis, vol_time
        )
    ####
    neural_trials = trial_label(ops, neural_trials)
    neural_trials = get_labels(neural_trials, first_epoch = 15, last_epoch = 15)
    # save the final data.
    print('Saving trial data as neural_trials')
    save_trials(ops, neural_trials)
    
    # creating neural_data
    print('===============================================')
    print('=============== Creating neural data ===============')
    print('===============================================')
    # processing dlc results.
    camera_time, camera_pupil = read_camera(ops, z_score = False)
    camera_time = correct_time_img_center(camera_time)
    camera_pupil = interp1d(time_pupil[:len(camera_pupil)], camera_pupil, bounds_error=False, fill_value=np.nan)(time_neuro)
    neural_data = create_neural_data(time_neuro, dff, 
                           spikes,
                           time_flir,
                           camera_time, camera_pupil,
                           start, end, stim)
    event_list = [
        'trial_push1', 'trial_push2',
        'trial_vis1', 'trial_vis2',
        'trial_wait2',
        'trial_reward', 'trial_reward_delay',
        'trial_retract1', 'trial_retract2',
        ]
    info_list = [
        'trial_types',
        'trial_outcome',
        'trial_block',
        'trial_block_len',
        'trial_pos_in_block',
        'trial_press_delay',
        ]
    neural_data = add_event_aray(neural_data, neural_trials, event_list, info_list)
    save_neural_data(ops, neural_data)
