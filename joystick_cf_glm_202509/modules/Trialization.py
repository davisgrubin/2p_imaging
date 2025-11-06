#!/usr/bin/env python3

import os
import h5py
import numpy as np

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_bpod_mat_data
from scipy.signal import savgol_filter


# detect the rising edge and falling edge of binary series.

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
        dff,event, stim, time_neuro,
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
            neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
            neural_trials[str(i)]['ca_event'] = event[:,start_idx_dff:end_idx_dff]
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
            'trial_push2'][i]-bpod_sess_data['trial_retract1'][i]
        neural_trials[str(i)]['trial_lick'][0,:] += neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_all_lick'] = bpod_sess_data[
            'trial_all_lick'][i] + neural_trials[str(i)]['vol_time'][0]
        neural_trials[str(i)]['trial_js_pos'] = bpod_sess_data[
            'trial_js_pos'][i]
        neural_trials[str(i)]['trial_js_time'] = bpod_sess_data[
            'trial_js_time'][i]
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
    
def detect_event(dff, percentile = 75, win_size = 5):
    events_area = np.zeros_like(dff)
    events_rise = np.zeros_like(dff)
    for neuron in range(dff.shape[0]):
        dff_smoothed = savgol_filter(dff[neuron], window_length=7, polyorder=3)
        dff_trace = dff_smoothed
        dff_dt = np.diff(dff_smoothed, prepend=dff_smoothed[0])
        percentile_exrtact = np.percentile(dff_trace, percentile)
        percentile_dt = np.percentile(dff_dt, 96)
        gaussian_area = percentile_exrtact*win_size
        for i in range(len(dff_trace)):
            # Extract window
            window = dff_trace[i:i+win_size]
            # Calculate area of this window
            window_area = np.sum(window)
            # Compare areas
            if window_area > gaussian_area:
                events_area[neuron, i] = 1
                
            if dff_dt[i] > percentile_dt:
                events_rise[neuron, i] = 1
        events_rise[neuron] = np.array(np.diff(events_rise[neuron], prepend=0)>0,dtype = int)
        events_area[neuron] = np.array(np.diff(events_area[neuron], prepend=0)>0,dtype = int)
                
    return events_area, events_rise

# save trial neural data.
def save_events(ops, name_area, name_rise, area, rise):
    f = h5py.File(os.path.join(ops['save_path0'], 'manual_qc_results', 'events.h5'), 'w')
    f[name_area] = area
    f[name_rise] = rise
    f.close()
    
    
def save_trials(
        ops,
        neural_trials
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # -- trial_id
    # ---- 1
    # ------ time
    # ------ stim
    # ------ dff
    # ---- 2
    # ...
    bpod_sess_data = read_bpod_mat_data(ops)
    np.abs(len(neural_trials)- len(bpod_sess_data['trial_types']))
    exclude_start = 5
    exclude_end = max(5,np.abs(len(neural_trials)- len(bpod_sess_data['trial_types'])))
    print(exclude_end)
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'w')
    grp = f.create_group('trial_id')
    print(len(neural_trials))
    for trial in range(len(neural_trials)):
        if trial > exclude_start and trial < len(neural_trials)-exclude_end:
            trial_group = grp.create_group(str(trial))
            for k in neural_trials[str(trial)].keys():
                trial_group[k] = neural_trials[str(trial)][k]
    f.close()


# main function for trialization

def run(ops):
    print('===============================================')
    print('=============== trial alignment ===============')
    print('===============================================')
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops, manual = True)
    dff_z_scored = (dff - dff.mean(axis=1, keepdims=True)) / dff.std(axis=1, keepdims=True)
    events_area, events_rise = detect_event(dff, percentile = 75, win_size = 5)
    save_events(ops, 'area','rise', events_area, events_rise)
    [vol_time, vol_start, vol_stim_vis, vol_img, 
     _, _, _, _, _] = read_raw_voltages(ops)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # stimulus alignment.
    print('Aligning stimulus to 2p frame')
    stim = align_stim(vol_time, time_neuro, vol_stim_vis, vol_stim_vis)
    # trial segmentation.
    print('Segmenting trials')
    start, end = get_trial_start_end(vol_time, vol_start)
    neural_trials = trial_split(
        start, end,
        dff_z_scored,events_area, stim, time_neuro,
        vol_stim_vis, vol_time)
    ####
    neural_trials = trial_label(ops, neural_trials)
    # save the final data.
    print('Saving trial data')
    save_trials(ops, neural_trials)
