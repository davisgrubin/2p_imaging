# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 13:29:30 2025

@author: saminnaji3
"""

import os
import h5py
import numpy as np

def read_camlog(file_path):
    '''
    Load a .camlog file and return three NumPy arrays:
        frame_id, timestamp, and third column (e.g., exposure or queue id).
    '''
    frame_id, tstamp, val = [], [], []
    camlog_path = os.path.join(file_path, 'camlog_file.camlog')
    if not os.path.isfile(camlog_path):
        raise FileNotFoundError(f'File not found: {camlog_path}')
    with open(camlog_path, 'r') as f:
        for line in f:
            line = line.strip()
            # skip header lines and empties
            if not line or line.startswith("#"):
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                frame_id.append(int(parts[0]))
                tstamp.append(float(parts[1]))
                val.append(float(parts[2]))
            except ValueError:
                # skip malformed numeric lines
                continue
    return np.array(frame_id, dtype=int), np.array(tstamp, dtype=float), np.array(val, dtype=float)

def read_raw_voltages(vol_path):
    '''
    Open raw_voltages.h5 from base_dir and return all required arrays.
    '''
    h5_path = os.path.join(vol_path, 'raw_voltages.h5')
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f'File not found: {h5_path}')

    with h5py.File(h5_path, 'r') as f:
        raw = f['raw']
        vol_time     = np.array(raw['vol_time'],     dtype=np.float32)
        vol_start    = np.array(raw['vol_start'],    dtype=np.int8)
        vol_stim_vis = np.array(raw['vol_stim_vis'], dtype=np.int8)
        vol_hifi     = np.array(raw['vol_hifi'],     dtype=np.int8)
        vol_img      = np.array(raw['vol_img'],      dtype=np.int8)
        vol_stim_aud = np.array(raw['vol_stim_aud'], dtype=np.float32)
        vol_flir     = np.array(raw['vol_flir'],     dtype=np.int8)
        vol_pmt      = np.array(raw['vol_pmt'],      dtype=np.int8)
        vol_led      = np.array(raw['vol_led'],      dtype=np.int8)
        
    return [vol_time, vol_start, vol_stim_vis, vol_img,
        vol_hifi, vol_stim_aud, vol_flir,
        vol_pmt, vol_led]

def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down

def find_flir_vol(vol_time, vol_flir):
    '''
    check if the flir voltage has been recorded or not
    '''
    if vol_flir[0] == 1:
        diff_vol = np.diff(vol_flir, prepend=1)
        idx_up = np.where(diff_vol == 1)[0]
        idx_down = np.where(diff_vol == -1)[0]
        if len(idx_down) > 0:
            idx_down = idx_down[1:]
    else:
        diff_vol = np.diff(vol_flir, prepend=0)
        idx_up = np.where(diff_vol == 1)[0]
        idx_down = np.where(diff_vol == -1)[0]
        
    
    return idx_up, idx_down

def camlog_times(timestamp, vol_time):
    latency = 5 # ms
    diff_timestamp = np.diff(timestamp)
    diff_timestamp = np.insert(diff_timestamp, 0, 0)
    time_approx = np.cumsum(1000*diff_timestamp)+latency
    
    #indices = np.abs(vol_time[:, None] - time_approx).argmin(axis=0)
    #time_flir = vol_time[indices]
    return time_approx

def nearest_time(time_approx, vol_time):
    ta = np.asarray(time_approx)
    vt = np.asarray(vol_time)

    order = np.argsort(vt)
    vt_sorted = vt[order]
    idx = np.searchsorted(vt_sorted, ta, side='left')
    idx = np.clip(idx, 1, len(vt_sorted) - 1)
    left_idx  = idx - 1
    right_idx = idx

    left  = vt_sorted[left_idx]
    right = vt_sorted[right_idx]
    pick_right = np.abs(ta - left) > np.abs(ta - right)
    best_sorted_idx = np.where(pick_right, right_idx, left_idx)
    best_orig_idx = order[best_sorted_idx]
    best_vals = vt[best_orig_idx]
    return best_vals

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

def run(vol_path, file_path):
    frame_id, timestamp, _ = read_camlog(file_path)
    [vol_time, vol_start, vol_stim_vis, vol_img,
        vol_hifi, vol_stim_aud, vol_flir,
        vol_pmt, vol_led] = read_raw_voltages(vol_path)
    time_img, _   = get_trigger_time(vol_time, vol_img)
    time_neuro = correct_time_img_center(time_img)
    
    idx_up, idx_down = find_flir_vol(vol_time, vol_flir)
    if len(idx_up) == 0:
        print('There is no flir voltage recording')
        print('using camlog file to align')
        time_approx = camlog_times(timestamp, vol_time)
        time_approx = time_approx + time_neuro[0]
        time_flir = nearest_time(time_approx, vol_time)
    else:
        print('using voltage recording file to align')
        time_flir = vol_time[idx_up]
    
    time_flir = correct_time_img_center(time_flir)
    return time_flir, time_neuro
    
    