#!/usr/bin/env python3

import os
import h5py
import numpy as np

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_bpod_mat_data


# remove trial start trigger voltage impulse.
def remove_start_impulse(vol_time, vol_stim_bin):
    min_duration = 100
    changes = np.diff(vol_stim_bin.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    if vol_stim_bin[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if vol_stim_bin[-1] == 1:
        end_indices = np.append(end_indices, len(vol_stim_bin))
    for start, end in zip(start_indices, end_indices):
        duration = vol_time[end-1] - vol_time[start]
        if duration < min_duration:
            vol_stim_bin[start:end] = 0
    return vol_stim_bin


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


# get stimulus sequence labels.

def get_stim_labels(bpod_sess_data, vol_time, vol_stim_bin):
    stim_time_up, stim_time_down = get_trigger_time(vol_time, vol_stim_bin)
    stim_labels = np.zeros((len(stim_time_up), 7))
    # row 0: stim start.
    # row 1: stim end.
    # row 2: img_seq_label.
    # row 3: normal_types.
    # row 4: fix_jitter_types.
    # row 5: oddball_types.
    # row 6: opto_types.
    stim_labels[:,0] = stim_time_up
    stim_labels[:,1] = stim_time_down
    stim_labels[:,2] = bpod_sess_data['img_seq_label']
    stim_labels[:,3] = bpod_sess_data['normal_types']
    stim_labels[:,4] = bpod_sess_data['fix_jitter_types']
    stim_labels[:,5] = bpod_sess_data['oddball_types']
    stim_labels[:,6] = bpod_sess_data['opto_types']
    return stim_labels


# align the stimulus sequence with fluorescence signal.

def align_stim(
        vol_time,
        time_neuro,
        vol_stim_bin,
        bpod_sess_data,
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
        stim[stim_start[i]:stim_end[i]] = bpod_sess_data['img_seq_label'][i]
    return stim


# compute when shutter is closed.

def get_pmt_close(vol_img, vol_pmt):
    # compute 2p image triggered time.
    diff_vol_img = np.diff(vol_img, prepend=0)
    vol_img_idx_up = np.where(diff_vol_img == 1)[0]
    # shutter is closed if vol_pmt == 1.
    pmt_close = np.zeros_like(vol_img_idx_up)
    pmt_close[vol_pmt[vol_img_idx_up] == 1] = 1
    pmt_close = pmt_close.astype('bool')
    return pmt_close
    

# interpolate dff when shutter is closed.

def dff_interpolation(dff, vol_time, vol_img, vol_pmt):
    plt.plot(vol_time, vol_led*2)
    plt.plot(vol_time, vol_pmt*3)
    plt.plot(vol_time, vol_img*4)
    plt.plot(time_img, pmt_close)
    plt.plot(time_img, dff[5,:])
    win = 5
    pmt_close = get_pmt_close(vol_img, vol_pmt)
    for i, close in enumerate(pmt_close):
        if close:
            # find elements wihtin window.
            neigh_idx = np.zeros_like(pmt_close).astype('bool')
            neigh_idx[i-win:i+win+1] = True
            # find elements where shutter is not closed.
            neigh_idx = neigh_idx * np.logical_not(pmt_close)
            # compute mean and replace.
            dff[:,i] = np.mean(dff[:,neigh_idx], axis=1)
    return dff


# save trial neural data.

def save_trials(
        ops,
        time_neuro, dff,
        vol_stim, vol_time,
        stim_labels
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # ---- time
    # ---- stim
    # ---- dff
    # ---- vol_stim
    # ---- vol_time
    # ---- stim_labels
    # ...
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'w')
    grp = f.create_group('neural_trials')
    grp['time'] = time_neuro
    grp['dff'] = dff
    grp['vol_stim'] = vol_stim
    grp['vol_time'] = vol_time
    grp['vol_time'] = vol_time
    grp['stim_labels'] = stim_labels
    f.close()


# main function for trialization

def run(ops):
    print('===============================================')
    print('=============== trial alignment ===============')
    print('===============================================')
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops)
    [vol_time, vol_start, vol_stim_vis, vol_img, 
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    vol_stim_vis = remove_start_impulse(vol_time, vol_stim_vis)
    bpod_sess_data = read_bpod_mat_data(ops)

    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    
    # stimulus sequence labeling
    stim_labels = get_stim_labels(bpod_sess_data, vol_time, vol_stim_vis)

    # save the final data.
    print('Saving trial data')
    save_trials(ops, time_neuro, dff, vol_stim_vis, vol_time, stim_labels)