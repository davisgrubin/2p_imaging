#!/usr/bin/env python3

import os
import h5py
import numpy as np

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_bpod_mat_data

# remove trial start trigger voltage impulse.
def remove_start_impulse(vol_time, vol_stim_vis):
    min_duration = 100
    changes = np.diff(vol_stim_vis.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    if vol_stim_vis[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if vol_stim_vis[-1] == 1:
        end_indices = np.append(end_indices, len(vol_stim_vis))
    for start, end in zip(start_indices, end_indices):
        duration = vol_time[end-1] - vol_time[start]
        if duration < min_duration:
            vol_stim_vis[start:end] = 0
    return vol_stim_vis

# correct beginning vol_stim_vis if not start from 0.
def correct_vol_start(vol_stim_vis):
    if vol_stim_vis[0] == 1:
        vol_stim_vis[:np.where(vol_stim_vis==0)[0][0]] = 0
    return vol_stim_vis

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

# find when bpod session timer start.
def get_session_start_time(vol_time, vol_start):
    time_up, _ = get_trigger_time(vol_time, vol_start)
    session_start_time = time_up[0]
    return session_start_time

# save trial neural data.
def save_trials(
        ops, time_neuro, dff, trial_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led
        ):
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # ---- time
    # ---- stim
    # ---- dff
    # ---- vol_stim
    # ---- vol_time
    # trial_labels.csv
    h5_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('neural_trials')
    grp['time']         = time_neuro
    grp['dff']          = dff
    grp['vol_time']     = vol_time
    grp['vol_stim_vis'] = vol_stim_vis
    grp['vol_stim_aud'] = vol_stim_aud
    grp['vol_flir']     = vol_flir
    grp['vol_pmt']      = vol_pmt
    grp['vol_led']      = vol_led
    f.close()
    trial_labels.to_csv(os.path.join(ops['save_path0'], 'trial_labels.csv'))

# main function for trialization.
def run(ops):
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops)
    [vol_time, vol_start, vol_stim_vis, vol_img,
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    vol_stim_vis = remove_start_impulse(vol_time, vol_stim_vis)
    vol_stim_vis = correct_vol_start(vol_stim_vis)
    session_start_time = get_session_start_time(vol_time, vol_start)
    trial_labels = read_bpod_mat_data(ops, session_start_time)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # save the final data.
    print('Saving trial data')
    save_trials(
        ops, time_neuro, dff, trial_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led)

'''

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(18, 6))
ax.plot(vol_time, vol_start)
ax.plot(vol_time, vol_stim_vis)
for ti in range(len(trial_labels)):
    for si in range(trial_labels['stim_seq'].to_numpy()[ti].shape[0]):
        ax.fill_between(trial_labels['stim_seq'].to_numpy()[ti][si,:], 0,1, color='dimgrey')

'''