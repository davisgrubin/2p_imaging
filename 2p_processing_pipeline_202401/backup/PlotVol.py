#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def thres_binary(
        data,
        thres
        ):
    data_bin = data.copy()
    data_bin[data_bin<thres] = 0
    data_bin[data_bin>thres] = 1
    return data_bin

def vol_to_binary(
        vol_start,
        vol_stim,
        vol_img
        ):
    vol_start_bin = thres_binary(vol_start, 1)
    vol_stim_bin  = thres_binary(vol_stim, 1)
    vol_img_bin   = thres_binary(vol_img, 1)
    return vol_start_bin, vol_stim_bin, vol_img_bin

filename = 'C:/Users/yhuang887/Downloads/LC4_C_Omi_20240405_t-358_Cycle00001_VoltageRecording_001.csv'
df_vol = pd.read_csv(filename, engine='python')
# column 0: time index in ms.
# column 1: trial start signal from bpod.
# column 2: stimulus signal from photodiode.
# column 3: BNC2 not in use.
# column 4: image trigger signal from 2p scope camera.
vol_time  = df_vol.iloc[:,0].to_numpy()
vol_start = df_vol.iloc[:,1].to_numpy()
vol_stim  = df_vol.iloc[:,2].to_numpy()
vol_img   = df_vol.iloc[:,4].to_numpy()
vol_start_bin, vol_stim_bin, vol_img_bin = vol_to_binary(
        vol_start, vol_stim, vol_img)

plt.plot(vol_time, vol_start, label='start')
plt.plot(vol_time, vol_stim, label='stim')

