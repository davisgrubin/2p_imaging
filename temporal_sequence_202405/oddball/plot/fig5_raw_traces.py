#!/usr/bin/env python3

import numpy as np
from plot.utils import get_sub_time_idx
from plot.utils import get_roi_label_color
from plot.utils import add_legend

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

# adjust layout for raw traces.
def adjust_layout_roi_raw_trace(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (s)')
    
# adjust layout for example traces.
def adjust_layout_example_trace(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_title('example traces')
    
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
    
# select example neural traces.
def plot_sess_example_traces(ax, list_dff, list_labels, list_vol, label_names):
    max_ms = 300000
    n = 20
    r = 0.75
    dff = list_dff[0]
    labels = list_labels[0]
    vol_img_bin = list_vol[0][3]
    vol_time = list_vol[0][0]
    _, _, c1, _ = get_roi_label_color([-1], 0)
    _, _, c2, _ = get_roi_label_color([1], 0)
    # find time indice.
    time_img = get_img_time(vol_time, vol_img_bin)
    start_time = np.max(time_img)/5.025520
    time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
    sub_time_img = time_img[time_img_idx]
    # select neurons.
    if len(label_names) == 1:
        idx = np.where(np.array(labels) == int(list(label_names.keys())[0]))[0]
        idx = np.random.choice(idx, n, replace=False)
    if len(label_names) == 2:
        idx1 = np.where(np.array(labels) == -1)[0]
        idx2 = np.where(np.array(labels) == 1)[0]
        n1 = int(r*n) if int(r*n) < len(idx1) else len(idx1)
        n2 = int((1-r)*n) if int((1-r)*n) < len(idx2) else len(idx2)
        idx1 = np.random.choice(idx1, n1, replace=False)
        idx2 = np.random.choice(idx2, n2, replace=False)
        idx = np.concatenate([idx1,idx2])
    # plot neural traces.
    sub_dff = dff[idx, :].copy()
    sub_dff = sub_dff[:, time_img_idx]
    scale = np.max(np.abs(sub_dff))
    for i in range(len(idx)):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color=[get_roi_label_color([i], 0)[2] for i in labels[idx]][i])
    adjust_layout_example_trace(ax)
    ax.set_ylim([-0.5*scale,(len(idx)+1)*scale])
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    add_legend(
        ax,
        [get_roi_label_color([int(k)], 0)[2] for k in label_names.keys()],
        [i[1] for i in label_names.items()],
        'upper right')
    
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
        adjust_layout_roi_raw_trace(axs[i])
        axs[i].set_title('raw trace of ROI # '+ str(roi_id).zfill(4))
        axs[i].set_ylim([lower - 0.1*(upper-lower),
                           upper + 0.1*(upper-lower)])
        axs[i].set_xlim([start, end])
        axs[i].set_xticks(i*max_ms + np.arange(0,max_ms/60000+1)*60000)
        axs[i].set_xticklabels((i*max_ms/60000 + np.arange(0,max_ms/60000+1))*60)