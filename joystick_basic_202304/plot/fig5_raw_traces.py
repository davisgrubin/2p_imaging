#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff


#%% utils


# rescale voltage recordings.

def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# read and process traces

def read_data(
        ops
        ):
    [vol_time, _, vol_stim_bin, vol_img_bin] = read_raw_voltages(ops)
    [labels, _, _, _, _] = read_masks(ops)
    dff = read_dff(ops)
    time_img = get_img_time(vol_time, vol_img_bin)
    return [labels, dff, time_img, vol_stim_bin, vol_time]


# find imaging trigger time stamps

def get_img_time(
        vol_time,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = vol_time[idx_up]
    return img_time


# get subsequence index with given start and end.

def get_sub_time_idx(
        time,
        start,
        end
        ):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx


# adjust layout for raw traces.

def adjust_layout_trace(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper left')
    

#%% ppc


# plot example traces for ppc.

def plot_ppc_examplt_traces(
        ax,
        dff, labels, vol_stim_bin, vol_img_bin, vol_time
        ):
    max_ms = 300000
    num_exc = 8
    num_inh = 2
    vol_stim = vol_stim_bin.copy()
    vol_stim[vol_stim!=0] = 1
    start_time = vol_time[vol_stim>0][5]
    time_img = get_img_time(vol_time, vol_img_bin)
    time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
    sub_time_img = time_img[time_img_idx]
    sub_dff_exc  = dff[labels==-1, :]
    sub_dff_exc  = sub_dff_exc[:np.min([num_exc, sub_dff_exc.shape[0]]), time_img_idx]
    sub_dff_inh  = dff[labels==-1, :]
    sub_dff_inh  = sub_dff_inh[:np.min([num_inh, sub_dff_inh.shape[0]]), time_img_idx]
    sub_dff = np.concatenate((sub_dff_exc, sub_dff_inh), axis=0)
    color_label  = np.zeros(num_exc+num_inh, dtype='int32') 
    color_label[num_exc:] = 1
    label = ['excitory', 'inhibitory']
    color = ['#A4CB9E', '#EDA1A4']
    scale = np.max(np.abs(sub_dff))*1.5
    for i in range(num_exc+num_inh):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color=color[color_label[i]],
            label=label[color_label[i]])
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    ax.set_title('example traces')
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[-1]]
    labels = [labels[0], labels[-1]]
    ax.legend(handles, labels, loc='upper right')


# ROI raw traces.

def plot_ppc_roi_raw_trace(
        axs, roi_id, max_ms,
        labels, dff, vol_img_bin, vol_stim_bin, vol_time,
        ):
    time_img = get_img_time(vol_time, vol_img_bin)
    upper = np.max(dff[roi_id, :])
    lower = np.min(dff[roi_id, :])
    color = ['seagreen', 'dodgerblue', 'coral']
    category = ['excitory', 'unsure', 'inhibitory']
    for i in range(len(axs)):
        start = i * max_ms
        end   = (i+1) * max_ms
        sub_vol_time_idx = get_sub_time_idx(vol_time, start, end)
        sub_time_img_idx = get_sub_time_idx(time_img, start, end)
        sub_vol_time     = vol_time[sub_vol_time_idx]
        sub_time_img     = time_img[sub_time_img_idx]
        sub_dff          = dff[roi_id, sub_time_img_idx]
        sub_vol_stim_bin = vol_stim_bin[sub_vol_time_idx]
        axs[i].fill_between(
            sub_vol_time,
            lower, upper,
            where=(sub_vol_stim_bin!=0),
            color='silver', step='mid', label='stim')
        axs[i].plot(
            sub_time_img,
            sub_dff,
            color=color[int(labels[roi_id]+1)],
            label=category[int(labels[roi_id]+1)],
            lw=0.5)
        adjust_layout_trace(axs[i])
        axs[i].set_title('raw trace of ROI # '+ str(roi_id).zfill(4))
        axs[i].set_ylim([lower - 0.1*(upper-lower),
                           upper + 0.1*(upper-lower)])
        axs[i].set_xlim([start, end])
        axs[i].set_xticks(i*max_ms + np.arange(0,max_ms/60000+1)*60000)
        axs[i].set_xticklabels((i*max_ms/60000 + np.arange(0,max_ms/60000+1))*60)

