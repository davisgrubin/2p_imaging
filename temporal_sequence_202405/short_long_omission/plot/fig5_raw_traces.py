#!/usr/bin/env python3

import numpy as np
from plot.utils import get_sub_time_idx
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_example_trace
from plot.utils import adjust_layout_raw_trace


# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# find imaging trigger time stamps
def get_img_time(
        vol_time,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = vol_time[idx_up]
    return img_time


# roi example traces.
def plot_roi_example_traces(
        ax,
        dff, labels, vol_img_bin, vol_time,
        roi_id
        ):
    max_ms = 100000
    time_img = get_img_time(vol_time, vol_img_bin)
    start_time = np.max(time_img)/4
    time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
    sub_time_img = time_img[time_img_idx]
    sub_dff = dff[roi_id, time_img_idx]
    _, _, color, _ = get_roi_label_color(labels, roi_id)
    ax.plot(sub_time_img, sub_dff, color=color)
    adjust_layout_example_trace(ax)
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    
    
# plot example traces for ppc.
def plot_VIPTD_G8_example_traces(
        ax,
        dff, labels, vol_img_bin, vol_time
        ):
    max_ms = 100000
    num_exc = 8
    num_inh = 2
    time_img = get_img_time(vol_time, vol_img_bin)
    start_time = np.max(time_img)/4
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
    _, _, c1, _ = get_roi_label_color([-1], 0)
    _, _, c2, _ = get_roi_label_color([1], 0)
    color = [c1, c2]
    scale = np.max(np.abs(sub_dff))*1.5
    for i in range(num_exc+num_inh):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color=color[color_label[i]],
            label=label[color_label[i]])
    adjust_layout_example_trace(ax)
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[-1]]
    labels = [labels[0], labels[-1]]
    ax.legend(handles, labels, loc='upper right')
        

# example traces for crbl.
def plot_L7G8_example_traces(
        ax,
        dff, vol_stim_bin, vol_img_bin, vol_time
        ):
    max_ms = 30000
    num_roi = 10
    vol_stim = vol_stim_bin.copy()
    vol_stim[vol_stim!=0] = 1
    time_img = get_img_time(vol_time, vol_img_bin)
    start_time = np.max(time_img)/4
    time_img = get_img_time(vol_time, vol_img_bin)
    time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
    sub_time_img = time_img[time_img_idx]
    sub_dff = dff[:np.min([num_roi, dff.shape[0]]), time_img_idx]
    scale = np.max(np.abs(sub_dff))*1.5
    for i in range(num_roi):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color='#A4CB9E')
    adjust_layout_example_trace(ax)
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])


# example traces for crbl.
def plot_VIPG8_example_traces(
        ax,
        dff, vol_stim_bin, vol_img_bin, vol_time
        ):
    max_ms = 30000
    num_roi = 10
    vol_stim = vol_stim_bin.copy()
    vol_stim[vol_stim!=0] = 1
    time_img = get_img_time(vol_time, vol_img_bin)
    start_time = np.max(time_img)/4
    time_img = get_img_time(vol_time, vol_img_bin)
    time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
    sub_time_img = time_img[time_img_idx]
    sub_dff = dff[:np.min([num_roi, dff.shape[0]]), time_img_idx]
    scale = np.max(np.abs(sub_dff))*1.5
    for i in range(num_roi):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color='#EDA1A4')
    adjust_layout_example_trace(ax)
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    

# ROI raw traces.
def plot_roi_raw_trace(
        axs, roi_id, max_ms,
        labels, dff,
        vol_img, vol_stim_vis, vol_pmt, vol_led, vol_time,
        ):
    time_img = get_img_time(vol_time, vol_img)
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
        sub_vol_stim_bin = vol_stim_vis[sub_vol_time_idx]
        axs[i].plot(
            sub_vol_time,
            sub_vol_stim_bin,
            color='grey',
            label='vis',
            lw=0.5)
        axs[i].plot(
            sub_time_img,
            sub_dff,
            color=color[int(labels[roi_id]+1)],
            label=category[int(labels[roi_id]+1)],
            lw=0.5)
        adjust_layout_raw_trace(axs[i])
        axs[i].set_title('raw trace of ROI # '+ str(roi_id).zfill(4))
        axs[i].set_ylim([lower - 0.1*(upper-lower),
                           upper + 0.1*(upper-lower)])
        axs[i].set_xlim([start, end])
        axs[i].set_xticks(i*max_ms + np.arange(0,max_ms/60000+1)*60000)
        axs[i].set_xticklabels((i*max_ms/60000 + np.arange(0,max_ms/60000+1))*60)