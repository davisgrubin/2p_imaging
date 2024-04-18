#!/usr/bin/env python3

import os
import h5py
import numpy as np

from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff
from postprocess.ReadResults import read_bpod_mat_data


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


# labeling grating sequence with bpod session data.

def label_stim_seq(vol_stim_bin, bpod_sess_data):
    try:
        # get edges for trial sequence in bpod session data.
        trial_seq = bpod_sess_data['trial_seq']
        diff_trial_seq = np.diff(trial_seq, prepend=0)
        diff_trial_seq = diff_trial_seq[diff_trial_seq!=0]
        # get edges for stimulus voltage recording.
        diff_vol_stim_bin = np.diff(vol_stim_bin, prepend=0)
        if len(diff_trial_seq) != len(np.where(diff_vol_stim_bin!=0)[0]):
            raise ValueError('Grating number is not consistent')
        else:
            label_stim = vol_stim_bin.copy()
            non_zero_indexes = np.where(diff_vol_stim_bin != 0)[0]
            for i in range(len(diff_trial_seq) // 2):
                stim_start_idx = non_zero_indexes[i * 2]
                stim_end_idx = non_zero_indexes[i * 2 + 1]
                label = diff_trial_seq[i * 2]
                label_stim[stim_start_idx: stim_end_idx] = label   
    except:
        print('Labeling failed and return with original voltage')
        label_stim = vol_stim_bin
    return label_stim


# align the stimulus sequence with fluorescence signal.

def align_stim(
        vol_time,
        time_neuro,
        vol_stim_bin,
        label_stim,
        ):
    # find the rising and falling time of stimulus.
    stim_time_up, stim_time_down = get_trigger_time(
        vol_time, vol_stim_bin)
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
        vol_start_bin,
        ):
    time_up, time_down = get_trigger_time(vol_time, vol_start_bin)
    # find the impulse start signal.
    time_start = []
    for i in range(len(time_up)):
        if time_down[i] - time_up[i] < 200:
            time_start.append(time_up[i])
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
        dff, stim, time_neuro,
        label_stim, vol_time,
        ):
    neural_trials = dict()
    for i in range(len(start)):
        neural_trials[str(i)] = dict()
        start_idx_dff = np.where(time_neuro > start[i])[0][0]
        end_idx_dff   = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
        neural_trials[str(i)]['time'] = time_neuro[start_idx_dff:end_idx_dff]
        neural_trials[str(i)]['stim'] = stim[start_idx_dff:end_idx_dff]
        neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
        start_idx_vol = np.where(vol_time > start[i])[0][0]
        end_idx_vol   = np.where(vol_time < end[i])[0][-1] if end[i] != -1 else -1
        neural_trials[str(i)]['vol_stim'] = label_stim[start_idx_vol:end_idx_vol]
        neural_trials[str(i)]['vol_time'] = vol_time[start_idx_vol:end_idx_vol]
    return neural_trials


# save trial neural data.

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
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'w')
    grp = f.create_group('trial_id')
    for trial in range(len(neural_trials)):
        trial_group = grp.create_group(str(trial))
        for k in neural_trials[str(trial)].keys():
            trial_group[k] = neural_trials[str(trial)][k]
    f.close()


# main function for trialization

def run(ops):
    print('===============================================')
    print('=============== trial alignment ===============')
    print('===============================================')
    try:
        print('Reading dff traces and voltage recordings')
        dff = read_dff(ops)
        [vol_time,
         vol_start_bin,
         vol_stim_bin,
         vol_img_bin] = read_raw_voltages(ops)
    
        print('Correcting 2p camera trigger time')
        # signal trigger time stamps.
        time_img, _   = get_trigger_time(vol_time, vol_img_bin)
        # correct imaging timing.
        time_neuro = correct_time_img_center(time_img)
        
        # label gratings with session data.
        print('Reading Bpod session data')
        bpod_sess_data = read_bpod_mat_data(ops)
        print('Labeling gratings')
        label_stim = label_stim_seq(vol_stim_bin, bpod_sess_data)
    
        # stimulus alignment.
        print('Aligning stimulus to 2p frame')
        stim = align_stim(vol_time, time_neuro, vol_stim_bin, label_stim)
    
        # trial segmentation.
        print('Segmenting trials')
        start, end = get_trial_start_end(vol_time, vol_start_bin)
        neural_trials = trial_split(
            start, end,
            dff, stim, time_neuro,
            label_stim, vol_time)
    
        # save the final data.
        print('Saving trial data')
        save_trials(ops, neural_trials)
    
    except:
        print('Trialization failed')
