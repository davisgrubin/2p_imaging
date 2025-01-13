#!/usr/bin/env python3

import numpy as np
from utils import get_mean_sem
from utils import get_sub_time_idx
from utils import get_ca_transient
from utils import get_roi_label_color
from utils import adjust_layout_neu
from utils import add_legend

# find imaging trigger time stamps
def get_img_time(
        vol_time,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = vol_time[idx_up]
    return img_time

# extract calcium transient average.
def get_ca_tran(dff, vol):
    timescale = 1.0
    l_frames = int(20*timescale)
    r_frames = int(150*timescale)
    time_img = get_img_time(vol[0], vol[3])
    # compute calcium transient time.
    ca_tran = get_ca_transient(dff)
    # extract dff traces around calcium transient.
    dff_ca_neu = []
    dff_ca_time = []
    for i in range(dff.shape[0]):
        ca_event_time = np.where(ca_tran[i,:]==1)[0]
        # find dff.
        cn = [dff[i,t-l_frames:t+r_frames].reshape(1,-1)
              for t in ca_event_time
              if t > l_frames and t < len(time_img)-r_frames]
        cn = np.concatenate(cn, axis=0)
        # find time.
        ct = [time_img[t-l_frames:t+r_frames].reshape(1,-1)-time_img[t]
              for t in ca_event_time
              if t > l_frames and t < len(time_img)-r_frames]
        ct = np.concatenate(ct, axis=0)
        ct = np.nanmean(ct, axis=0)
        # collect.
        dff_ca_neu.append(cn)
        dff_ca_time.append(ct)
    dff_ca_time = np.concatenate([t.reshape(1,-1) for t in dff_ca_time], axis=0)
    dff_ca_time = np.nanmean(dff_ca_time, axis=0)
    return dff_ca_neu, dff_ca_time

# adjust layout for raw traces.
def adjust_layout_roi_raw_trace(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (s)')

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
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])

# select example neural traces.
def plot_sess_example_traces(ax, dff, labels, vol, label_names):
    max_ms = 150000
    n = 20
    r = 0.75
    _, _, c1, _ = get_roi_label_color([-1], 0)
    _, _, c2, _ = get_roi_label_color([1], 0)
    # find time indice.
    def get_time_idx():
        time_img = get_img_time(vol[0], vol[3])
        start_time = np.max(time_img)/5.19961106
        time_img_idx = get_sub_time_idx(time_img, start_time, start_time+max_ms)
        sub_time_img = time_img[time_img_idx]
        return time_img_idx, sub_time_img
    # find active neurons with most calcium transients.
    def get_active_neuron(dff, labels, cate, n):
        # correct number.
        idx = np.where(np.array(labels) == cate)[0]
        n = n if n < len(idx) else len(idx)
        # compute calcium events.
        ca_tran = get_ca_transient(dff)
        # get the total number of events for each neurons.
        n_ca = np.nansum(ca_tran, axis=1)
        # find the best neurons.
        act_idx = np.argsort(np.where(labels==cate, n_ca, 0))[-n:]
        return act_idx
    # pick neuron indice.
    def get_idx(dff):
        if len(label_names) == 1:
            cate = int(list(label_names.keys())[-1])
            idx = get_active_neuron(dff, labels, cate, n)
        if len(label_names) == 2:
            idx1 = get_active_neuron(dff, labels, -1, int(r*n))
            idx2 = get_active_neuron(dff, labels, 1,  int((1-r)*n))
            idx = np.concatenate([idx1,idx2])
        return idx
    # main.
    time_img_idx, sub_time_img = get_time_idx()
    act_idx = get_idx(dff)
    sub_dff = dff[:, time_img_idx].copy()
    sub_dff = sub_dff[act_idx, :].copy()
    # plot neural traces.
    scale = np.max(np.abs(sub_dff))
    for i in range(len(act_idx)):
        ax.plot(
            sub_time_img, sub_dff[i,:] + i * scale,
            color=[get_roi_label_color([i], 0)[2] for i in labels[act_idx]][i])
    # adjust layout.
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_title('example traces')
    ax.set_ylim([-0.5*scale,len(act_idx)*scale])
    ax.set_xlim([np.min(sub_time_img), np.max(sub_time_img)])
    add_legend(
        ax,
        [get_roi_label_color([int(k)], 0)[2] for k in label_names.keys()],
        [i[1] for i in label_names.items()], None, None,
        'upper right')

# calcium transient grand average.
def plot_ca_tran(axs, dff, labels, vol, label_names):
    # find the timing or critical points.
    def get_half_decay_time(m):
        start = np.argmin(m)
        peak = np.argmax(m)
        half = peak + np.where(m[peak:]<np.max(m)/2)[0][0]
        return start, peak, half
    # plotting function.
    def plot_cate(ax, cate):
        color0, _, color2, _ = get_roi_label_color([cate], 0)
        label_name = label_names[str(cate)]
        # get category.
        dff_ca_cate = np.array(dff_ca_neu,dtype='object')[labels==cate].copy().tolist()
        # average across trials.
        dff_ca_cate = [get_mean_sem(d)[0].reshape(1,-1) for d in dff_ca_cate]
        # average across neurons.
        dff_ca_cate = np.concatenate(dff_ca_cate, axis=0)
        m,s = get_mean_sem(dff_ca_cate)
        # compute bounds.
        upper = np.nanmax(m) + np.nanmax(s)
        lower = np.nanmin(m) - np.nanmax(s)
        # mark times.
        start, peak, half = get_half_decay_time(m)
        ax.axvline(dff_ca_time[start], color=color0, lw=1, linestyle='--')
        ax.axvline(dff_ca_time[peak],  color=color0, lw=1, linestyle='--')
        ax.axvline(dff_ca_time[half],  color=color0, lw=1, linestyle='--')
        # plot results.
        ax.plot(dff_ca_time, m, color=color2)
        ax.fill_between(dff_ca_time, m - s, m + s, color=color2, alpha=0.2)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim([np.min(dff_ca_time), np.max(dff_ca_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since calcium transient triggered (ms)')
        ax.set_title(f'calcium transient average \n {label_name}')
    # main.
    dff_ca_neu, dff_ca_time = get_ca_tran(dff, vol)
    try:
        cate = -1
        plot_cate(axs[0], cate)
    except: pass
    try:
        cate = 1
        plot_cate(axs[1], cate)
    except: pass

# calcium transient half decay time distribution.
def plot_ca_tran_half_dist(axs, dff, labels, vol, label_names):
    # find the half decay time.
    def get_half_decay_time(m):
        peak = np.argmax(m)
        half = np.where(m[peak:]<np.max(m)/2)[0]
        if len(half) > 0:
            half = dff_ca_time[peak+half[0]] - dff_ca_time[peak]
        else:
            half = np.nan
        return half
    # plotting function.
    def plot_cate(ax, cate):
        t_range = [0,4000]
        color0, _, color2, _ = get_roi_label_color([cate], 0)
        label_name = label_names[str(cate)]
        # get category.
        dff_ca_cate = np.array(dff_ca_neu,dtype='object')[labels==cate].copy().tolist()
        # average across trials.
        dff_ca_cate = [get_mean_sem(d)[0].reshape(1,-1) for d in dff_ca_cate]
        # average across neurons.
        dff_ca_cate = np.concatenate(dff_ca_cate, axis=0)
        # compute half decay time.
        half = np.apply_along_axis(get_half_decay_time, 1, dff_ca_cate)
        # plot distribution.
        ax.hist(
            half, bins=200, range=t_range,
            weights=np.ones(len(half))/len(half),
            align='mid', color=color2,)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('half decay time (ms)')
        ax.set_ylabel('fraction')
        ax.set_xlim(t_range)
        ax.set_title(f'half decay time distribution \n {label_name}')
    # main.
    dff_ca_neu, dff_ca_time = get_ca_tran(dff, vol)
    try:
        cate = -1
        plot_cate(axs[0], cate)
    except: pass
    try:
        cate = 1
        plot_cate(axs[1], cate)
    except: pass

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
