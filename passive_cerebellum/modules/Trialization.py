#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.interpolate import interp1d

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_spikes
from modules.ReadResults import read_camera
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
    # import matplotlib.pyplot as plt
    # plt.plot(diff_vol[50000:100000])
    # plt.show()
    print('diff_vol',diff_vol)
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
def get_stim_labels(bpod_sess_data, vol_time, vol_stim_vis, trialized = 1):
    #print('this one')
    stim_time_up, stim_time_down = get_trigger_time(vol_time, vol_stim_vis)
    #end = min(len(bpod_sess_data['img_seq_label']), len(stim_time_up))
    print(len(bpod_sess_data['img_seq_label']), len(stim_time_up))
    if bpod_sess_data['img_seq_label'][-1] == -1:
        stim_time_up = stim_time_up[:-1]
        stim_time_down = stim_time_down[:-1]
    stim_labels = np.zeros((len(stim_time_up), 9))
    # row 0: stim start.
    # row 1: stim end.
    # row 2: img_seq_label.
    # row 3: standard_types.
    # row 4: fix_jitter_types.
    # row 5: oddball_types.
    # row 6: random_types.
    # row 7: opto_types.
    # row 8: sequence_number.
    stim_labels[:,0] = stim_time_up[:len(stim_time_up)]
    stim_labels[:,1] = stim_time_down[:len(stim_time_up)]
    if not len(bpod_sess_data['img_seq_label']) == len(stim_time_up):
        print('not same stim and recording len')
    if trialized:
        L = len(stim_time_up)   # your target length
        base = np.array([0,1,2,3])
        stim_labels[:,8] = np.resize(base, L)
        
    #print(stim_labels[1:160,0]-stim_labels[:159,1])
    
    
    stim_labels[:,2] = bpod_sess_data['img_seq_label'][:len(stim_time_up)]
    stim_labels[:,3] = bpod_sess_data['standard_types'][:len(stim_time_up)]
    stim_labels[:,4] = bpod_sess_data['fix_jitter_types'][:len(stim_time_up)]
    stim_labels[:,5] = bpod_sess_data['oddball_types'][:len(stim_time_up)]
    stim_labels[:,6] = bpod_sess_data['random_types'][:len(stim_time_up)]
    stim_labels[:,7] = bpod_sess_data['opto_types'][:len(stim_time_up)]
    
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    for trial in range(len(isi)):
        #print(stim_labels[trial,7])
        #if isi[trial] > 3000:
            #print(stim_labels[trial, 8])
        if np.abs(isi[trial] - 500) < 70 and stim_labels[trial,4] == 0:
            stim_labels[trial,2] = -1
            # stim_labels[trial-1,2] = -1
            # stim_labels[trial+1,2] = -1
            # stim_labels[trial+2,2] = -1
            #print(img_seq_label_final[trial])
            #print('ISIs: ', isi[trial-1:trial+3])
            #print(stim_labels[trial, 8])
            stim_labels[trial,5] = 0
            stim_labels[trial-1,5] = 0
            stim_labels[trial+1,5] = 0
            stim_labels[trial+2,5] = 0
        elif np.abs(isi[trial] - 2500) < 70 and stim_labels[trial,4] == 0:
            stim_labels[trial,2] = -1
            #print('ISIs: ', isi[trial-1:trial+3])
            #print(stim_labels[trial, 8])
            # stim_labels[trial-1,2] = -1
            # stim_labels[trial+1,2] = -1
            # stim_labels[trial+2,2] = -1
            #print(img_seq_label_final[trial])
            stim_labels[trial,5] = 1
            stim_labels[trial-1,5] = 1
            stim_labels[trial+1,5] = 1
            stim_labels[trial+2,5] = 1
        if np.abs(isi[trial] - 500) < 90 and stim_labels[trial,4] == 1:
            stim_labels[trial,2] = -1
            #print('short',trial)
            stim_labels[trial,5] = 0
        elif np.abs(isi[trial] - 2500) < 70 and stim_labels[trial,4] == 1:
            stim_labels[trial,2] = -1
            #print('long', trial)
            stim_labels[trial,5] = 1
    #print('random_type: ', stim_labels[25:50, 6])
    #print('seq_num: ', stim_labels[25:50, 8])
    #print('isi: ', isi[25:50])
    
    # this is for exeptions
    #stim_labels = stim_labels[1:, :]
    
    return stim_labels
    
# save trial neural data.
def save_trials(
        ops, time_neuro, dff, stim_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led,
        camera_time, camera_pupil,
        spikes
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
    h5_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('neural_trials')
    grp['time']         = time_neuro
    grp['dff']          = dff
    grp['spikes']          = spikes
    grp['stim_labels']  = stim_labels
    grp['vol_time']     = vol_time
    grp['vol_stim_vis'] = vol_stim_vis
    grp['vol_stim_aud'] = vol_stim_aud
    grp['vol_flir']     = vol_flir
    grp['vol_pmt']      = vol_pmt
    grp['vol_led']      = vol_led
    grp['camera_time']  = camera_time
    grp['camera_pupil'] = camera_pupil
    f.close()
    
def add_spikes(ops):
    h5_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    spikes = read_spikes(ops)
    if not os.path.exists(h5_path):
        print("File not found!")
        return

    # Use 'r+' to read/write without truncating the file
    with h5py.File(h5_path, 'r+') as f:
        # Access the existing group
        if 'neural_trials' in f:
            grp = f['neural_trials']
            
            # Check if 'spikes' already exists to avoid errors
            if 'spikes' in grp:
                print("Spikes dataset already exists. Overwriting...")
                del grp['spikes']
            
            # Create the new dataset
            grp.create_dataset('spikes', data=spikes)
            print("Spikes successfully added.")
        else:
            print("Group 'neural_trials' not found in file.")
            
# main function for trialization.
def run(ops, aud = 0):
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops)
    spikes = read_spikes(ops)
    [vol_time, vol_start, vol_stim_vis, vol_img,
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    if aud:
        vol_stim =  vol_hifi
        print('Audio Stim Session')
    else:
        vol_stim =  vol_stim_vis
    vol_stim_vis = remove_start_impulse(vol_time, vol_stim)
    vol_stim_vis = correct_vol_start(vol_stim_vis)
    bpod_sess_data = read_bpod_mat_data(ops, aud = aud)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # stimulus sequence labeling.
    stim_labels = get_stim_labels(bpod_sess_data, vol_time, vol_stim_vis)
    # processing dlc results.
    camera_time, camera_pupil = read_camera(ops)
    camera_time = correct_time_img_center(camera_time)
    camera_pupil = interp1d(camera_time, camera_pupil, bounds_error=False, fill_value=np.nan)(time_neuro)
    # save the final data.
    print('Saving trial data')
    save_trials(
        ops, time_neuro, dff, stim_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led,
        camera_time, camera_pupil, spikes)
    