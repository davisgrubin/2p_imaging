#!/usr/bin/env python3

import gc
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from tifffile import imread
from tifffile import imwrite


#%% tiff video

# find the target filenames in ops['data_path'].

def list_filenames(
        ops
        ):
    # ch1: SESSION_Cycle00001_Ch1_000NUM.ome.tif.
    # ch2: SESSION_Cycle00001_Ch2_000NUM.ome.tif.
    ch1_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch1' in f and '.tif' in f]
    ch2_files = [f for f in os.listdir(ops['data_path'])
                 if 'Ch2' in f and '.tif' in f]
    # sort the filename based on the recording partition number.
    ch1_files.sort()
    ch2_files.sort()
    return ch1_files, ch2_files


# read the tif files given the filename list.

def read_tif_to_np(
        ops,
        ch_files
        ):
    # no channel data found.
    if len(ch_files) == 0:
        ch_data = None
    # read into array.
    else:
        ch_data = np.empty((0, ops['Lx'], ops['Ly']), dtype='float32')
        for f in tqdm(ch_files):
            data = imread(os.path.join(ops['data_path'], f))
            ch_data = np.concatenate((ch_data, data), axis=0)
            # release memory.
            del data
            gc.collect()
    return ch_data


# 




#%% voltage


# read the voltage recording file.

def read_vol_to_np(
        ops,
        ):
    # voltage: SESSION_Cycle00001_VoltageRecording_000NUM.csv.
    vol_record = [f for f in os.listdir(ops['data_path'])
                  if 'VoltageRecording' in f and '.csv' in f]
    df_vol = pd.read_csv(
        os.path.join(ops['data_path'], vol_record[0]),
        engine='python')
    # column 0: time index in ms.
    # column 1: trial start signal from bpod.
    # column 2: stimulus signal from photodiode.
    # column 3: BNC2 not in use.
    # column 4: image trigger signal from 2p scope camera.
    vol_time  = df_vol.iloc[:,0].to_numpy()
    vol_start = df_vol.iloc[:,1].to_numpy()
    vol_stim  = df_vol.iloc[:,2].to_numpy()
    vol_img   = df_vol.iloc[:,4].to_numpy()
    return vol_time, vol_start, vol_stim, vol_img


# threshold the continuous voltage recordings to 01 series.

def thres_binary(
        data,
        thres
        ):
    data_bin = data.copy()
    data_bin[data_bin<thres] = 0
    data_bin[data_bin>thres] = 1
    return data_bin


# convert all voltage recordings to binary series.

def vol_to_binary(
        vol_start,
        vol_stim,
        vol_img
        ):
    vol_start_bin = thres_binary(vol_start, 1)
    vol_stim_bin  = thres_binary(vol_stim, 1)
    vol_img_bin   = thres_binary(vol_img, 1)
    return vol_start_bin, vol_stim_bin, vol_img_bin


# save voltage data.

def save_vol(
        ops,
        vol_time, vol_start_bin, vol_stim_bin, vol_img_bin,
        ):
    # file structure:
    # ops['save_path0'] / raw_voltages.h5
    # -- raw
    # ---- vol_time
    # ---- vol_start_bin
    # ---- vol_stim_bin
    # ---- vol_img_bin
    f = h5py.File(os.path.join(
        ops['save_path0'], 'raw_voltages.h5'), 'w')
    grp = f.create_group('raw')
    grp['vol_time']      = vol_time
    grp['vol_start_bin'] = vol_start_bin
    grp['vol_stim_bin']  = vol_stim_bin
    grp['vol_img_bin']   = vol_img_bin
    f.close()
    
    
#%% main function.

def run(ops):

    print('===============================================')
    print('========== read and merge video data ==========')
    print('===============================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ch1_files, ch2_files = list_filenames(ops)
    print('Found {} files for channel 1'.format(len(ch1_files)))
    print('Found {} files for channel 2'.format(len(ch2_files)))

    print('Reading channel 1 data')
    print(ch1_files)
    ch1_data = read_tif_to_np(ops, ch1_files)

    print('Reading channel 2 data')
    print(ch2_files)
    ch2_data = read_tif_to_np(ops, ch2_files)
    
    
    
    print('Processing voltage recordings')
    try:
        vol_time, vol_start, vol_stim, vol_img = read_vol_to_np(ops)
        vol_start_bin, vol_stim_bin, vol_img_bin = vol_to_binary(
            vol_start, vol_stim, vol_img)
        save_vol(ops, vol_time, vol_start_bin, vol_stim_bin, vol_img_bin)
    except:
        print('Valid voltage recordings csv file not found')
        
    return [ch1_data, ch2_data]
