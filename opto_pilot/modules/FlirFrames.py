# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:26:21 2025

@author: saminnaji3
"""

import os
import h5py
import numpy as np

from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_bpod_mat_data

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




def run(ops):
    print('===============================================')
    print('=============== flir frames timing ===============')
    print('===============================================')
    print('Reading voltage recordings')
    [vol_time, 
     vol_start, 
     vol_stim_vis, 
     vol_img, 
     vol_hifi, 
     vol_stim_aud, 
     vol_flir,
     vol_pmt, 
     vol_led] = read_raw_voltages(ops)
    
    # signal trigger time stamps.
    time_img, _   = get_trigger_time(vol_time, vol_flir)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    print('The number of 2P frames = ' + str(len(time_neuro)))
    return time_neuro
