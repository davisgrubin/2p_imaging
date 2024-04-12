#!/usr/bin/env python3

import os
import h5py
import numpy as np

from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff


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
        vol_stim_bin,
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
        stim[stim_start[i]:stim_end[i]] = 1
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
        dff,
        stim,
        time_neuro,
        start, end
        ):
    neural_trials = dict()
    for i in range(len(start)):
        # find start and end timing index.
        start_idx = np.where(time_neuro > start[i])[0][0]
        end_idx = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
        # save time and stimulus for each trial.
        neural_trials[str(i)] = dict()
        neural_trials[str(i)]['time'] = time_neuro[start_idx:end_idx]
        neural_trials[str(i)]['stim'] = stim[start_idx:end_idx]
        # save traces for each trial.
        neural_trials[str(i)]['dff'] = dff[:,start_idx:end_idx]
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


# tentative function to hard code trial types for passive sessions.

def add_jitter_flag(ops, neural_trials):
    from sklearn.mixture import GaussianMixture
    def frame_dur(stim, time):
        diff_stim = np.diff(stim, prepend=0)
        idx_up   = np.where(diff_stim == 1)[0]
        idx_down = np.where(diff_stim == -1)[0]
        dur_high = time[idx_down] - time[idx_up]
        dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
        return [dur_high, dur_low]
    def get_mean_std(isi):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(isi.reshape(-1,1))
        std = np.mean(np.sqrt(gmm.covariances_.flatten()))
        return std
    thres = 25
    jitter_flag = []
    for i in range(len(neural_trials)):
        stim = neural_trials[str(i)]['stim']
        time = neural_trials[str(i)]['time']
        [_, isi] = frame_dur(stim, time)
        std = get_mean_std(isi)
        jitter_flag.append(std)
    jitter_flag = np.array(jitter_flag)
    jitter_flag[jitter_flag<thres] = 0
    jitter_flag[jitter_flag>thres] = 1
    np.save(os.path.join(ops['save_path0'], 'jitter_flag.npy'), jitter_flag)


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
    
        # stimulus alignment.
        print('Aligning stimulus to 2p frame')
        stim = align_stim(vol_time, time_neuro, vol_stim_bin)
    
        # trial segmentation.
        print('Segmenting trials')
        start, end = get_trial_start_end(vol_time, vol_start_bin)
        neural_trials = trial_split(dff, stim, time_neuro, start, end)
    
        # save the final data.
        print('Saving trial data')
        save_trials(ops, neural_trials)
    
        # add trial type.
        print('Getting trial meta infomation')
        add_jitter_flag(ops, neural_trials)
    
    except:
        print('Trialization failed')
