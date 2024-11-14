#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.nanmin(data) ) / (np.nanmax(data) - np.nanmin(data))
    data = data * (upper - lower) + lower
    return data

# bin data for histogram.
def get_bin_stat(data, win, bin_size):
    min_samples = 5
    bins = np.arange(win[0], win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    bin_idx = np.digitize(data, bins) - 1
    bin_num = []
    bin_mean = []
    bin_sem = []
    for i in range(len(bins) - 1):
        bin_samples = data[(bin_idx == i)]
        bin_num.append(len(bin_samples))
        if len(bin_samples) > min_samples:
            m, s = get_mean_sem(bin_samples.reshape(-1,1))
        else:
            m = np.array([np.nan])
            s = np.array([np.nan])
        bin_mean.append(m)
        bin_sem.append(s)
    bin_num  = np.array(bin_num).reshape(-1)
    bin_mean = np.array(bin_mean).reshape(-1)
    bin_sem  = np.array(bin_sem).reshape(-1)
    non_nan    = (1-np.isnan(bin_mean)).astype('bool')
    bin_center = bin_center[non_nan]
    bin_num    = bin_num[non_nan]/len(data)
    bin_mean   = bin_mean[non_nan]
    bin_sem    = bin_sem[non_nan]
    return bin_center, bin_num, bin_mean, bin_sem

# compute baseline within given time window.
def get_base_mean_win(neu_seq, neu_time, c_time, win_base):
    pct = 30
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, win_base[0], win_base[1])
    neu = neu_seq[:, l_idx:r_idx].copy().reshape(-1)
    neu = neu[neu<np.nanpercentile(neu, pct)]
    mean_base = np.nanmean(neu)
    return mean_base

# compute mean and sem across trials for mean df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time):
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win_mean = np.nanmean(neu_seq[:, l_idx:r_idx], axis=1)
    neu_mean = np.nanmean(neu_win_mean)
    std = np.nanstd(neu_win_mean, axis=0)
    count = np.nansum(~np.isnan(neu_win_mean), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu_mean, neu_sem

# compute mean and sem for 3d array data
def get_mean_sem(data, zero_start=False):
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    m = m-m[0] if zero_start else m
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# get subsequence index with given start and end.
def get_sub_time_idx(time, start, end):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx

# normalize and apply colormap
def apply_colormap(data, cmap):
    if data.shape[1] == 0:
        return np.zeros((0, data.shape[1], 3))
    for i in range(data.shape[0]):
        data[i,:] = norm01(data[i,:])
    data_heatmap = cmap(data)
    data_heatmap = data_heatmap[..., :3]
    return data_heatmap

# mark stim after omi as outlier.
def exclude_post_odd_stim(stim_labels):
    stim_labels_mark = stim_labels.copy()
    idx_post = np.diff(stim_labels_mark[:,2]==-1, prepend=0)
    idx_post[idx_post==1] = 0
    idx_post[idx_post==-1] = 1
    idx_post = idx_post.astype('bool')
    stim_labels_mark[idx_post,2] = -1
    return stim_labels_mark

# find trials based on stim_labels.
def pick_trial(
        stim_labels,
        img_seq_label,
        normal_types,
        fix_jitter_types,
        oddball_types,
        opto_types):
    idx1 = np.isin(stim_labels[:,2], img_seq_label)    if img_seq_label    else np.ones_like(stim_labels[:,2])
    idx2 = np.isin(stim_labels[:,3], normal_types)     if normal_types     else np.ones_like(stim_labels[:,3])
    idx3 = np.isin(stim_labels[:,4], fix_jitter_types) if fix_jitter_types else np.ones_like(stim_labels[:,4])
    idx4 = np.isin(stim_labels[:,5], oddball_types)    if oddball_types    else np.ones_like(stim_labels[:,5])
    idx5 = np.isin(stim_labels[:,6], opto_types)       if opto_types       else np.ones_like(stim_labels[:,6])
    idx = (idx1*idx2*idx3*idx4*idx5).astype('bool')
    return idx

# for multi session settings find trials based on stim_labels and trial avergae.
def get_multi_sess_neu_trial_average(
        list_stim_labels,
        neu_cate,
        list_stim_seq,
        list_stim_value,
        list_pre_isi,
        trial_idx=None,
        trial_param=None,
        mean_sem=True,
        ):
    neu = []
    stim_seq = []
    stim_value = []
    pre_isi = []
    # use stim_labels to find trials.
    if trial_param != None and trial_idx == None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4])
            neu.append(neu_cate[i][idx,:,:])
            stim_seq.append(list_stim_seq[i][idx,:,:])
            stim_value.append(list_stim_value[i][idx,:])
            pre_isi.append(list_pre_isi[i][idx])
    # use given idx to find trials.
    if trial_param == None and trial_idx != None:
        for i in range(len(neu_cate)):
            neu.append(neu_cate[i][trial_idx[i],:,:])
            stim_seq.append(list_stim_seq[i][trial_idx[i],:,:])
            stim_value.append(list_stim_value[i][trial_idx[i],:])
            pre_isi.append(list_pre_isi[i][trial_idx[i]])
    # use both.
    if trial_param != None and trial_idx != None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4])
            neu.append(neu_cate[i][trial_idx[i]*idx,:,:])
            stim_seq.append(list_stim_seq[i][trial_idx[i]*idx,:,:])
            stim_value.append(list_stim_value[i][trial_idx[i]*idx,:])
            pre_isi.append(list_pre_isi[i][trial_idx[i]*idx])
    # compute trial average and concatenate.
    if mean_sem:
        mean = [np.nanmean(n, axis=0) for n in neu]
        sem  = [np.nanstd(n, axis=0)/np.sqrt(np.sum(~np.isnan(n), axis=0)) for n in neu]    
        mean = np.concatenate(mean, axis=0)
        sem  = np.concatenate(sem, axis=0)
        stim_seq   = np.mean(np.concatenate(stim_seq, axis=0),axis=0)
        stim_value = np.mean(np.concatenate(stim_value, axis=0),axis=0)
        return mean, sem, stim_seq, stim_value, None
    # return single trial response.
    else:
        return neu, stim_seq, stim_value, pre_isi

# find index for each epoch.
def get_epoch_idx(stim_labels):
    num_early_trials = 50
    switch_idx = np.where(np.diff(stim_labels[:,3], prepend=1-stim_labels[:,3][0])!=0)[0]
    epoch_early = np.zeros_like(stim_labels[:,3], dtype='bool')
    for start in switch_idx:
        epoch_early[start:start+num_early_trials] = True
    epoch_late = ~epoch_early
    return epoch_early, epoch_late

# find expected isi.
def get_expect_time(stim_labels):
    idx_short = (stim_labels[:,2]>1)*(stim_labels[:,3]==0)
    expect_short = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_short = np.median(expect_short[idx_short[:-1]])
    idx_long = (stim_labels[:,2]>1)*(stim_labels[:,3]==1)
    expect_long = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_long = np.median(expect_long[idx_long[:-1]])
    return expect_short, expect_long

# get index for short/long or pre/post.
def get_odd_stim_prepost_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,3]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,3]==1)
    idx_post_short = np.zeros_like(idx_pre_short)
    idx_post_short[1:] = idx_pre_short[:-1]
    idx_post_long = np.zeros_like(idx_pre_long)
    idx_post_long[1:] = idx_pre_long[:-1]
    idx_pre_short[-1] = False
    idx_pre_long[-1] = False
    return idx_pre_short, idx_pre_long, idx_post_short, idx_post_long

# get index for pre/post image change.
def get_change_prepost_idx(stim_labels):
    idx_pre = np.diff(stim_labels[:,2]<-1, append=0)
    idx_pre[idx_pre==-1] = 0
    idx_pre = idx_pre.astype('bool')
    idx_post = stim_labels[:,2]<-1
    return idx_pre, idx_post

# get ROI color from label.
def get_roi_label_color(labels, roi_id):
    if labels[roi_id] == -1:
        cate = 'excitory'
        color1 = 'grey'
        color2 = 'dodgerblue'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'dodgerblue', 'black'])
    if labels[roi_id] == 0:
        cate = 'unsure'
        color1 = 'grey'
        color2 = 'mediumseagreen'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'mediumseagreen', 'black'])
    if labels[roi_id] == 1:
        cate = 'inhibitory'
        color1 = 'grey'
        color2 = 'hotpink'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'hotpink', 'black'])
    return cate, color1, color2, cmap

# adjust layout for grand average neural traces.
def adjust_layout_neu(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper right')

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])

# adjust layout for example traces.
def adjust_layout_example_trace(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)')
    ax.set_title('example traces')

# adjust layout for raw traces.
def adjust_layout_raw_trace(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper left')


class utils:

    def __init__(self):
        self.min_num_trial = 5
        self.color_single = 'grey'
        self.color_isi = ['blue','red']
        self.color_epoch = ['mediumseagreen', 'coral']

    def plot_mean_sem(self, ax, t, m, s, c, l, a=1.0):
        ax.plot(t, m, color=c, label=l, alpha=a)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])

    def plot_vol(self, ax, st, sv, c, u, l):
        v = np.mean(sv, axis=0)
        v = rescale(v, u, l)
        ax.plot(st, v, color=c, lw=0.5, linestyle=':')

    def plot_heatmap_neuron(
            self, ax, neu_seq, neu_time, neu_seq_sort,
            win_sort, labels, s, colorbar=False):
        win_conv = 5
        if len(neu_seq) > 0:
            _, _, _, cmap_exc = get_roi_label_color([-1], 0)
            _, _, _, cmap_inh = get_roi_label_color([1], 0)
            zero = np.searchsorted(neu_time, 0)
            # exclude nan.
            neu_idx = ~np.isnan(np.sum(neu_seq,axis=1))
            neu_seq = neu_seq[neu_idx,:].copy()
            neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
            labels = labels[neu_idx].copy()
            s = s[neu_idx].copy()
            # smooth the values in the sorting window.
            l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_sort[0], win_sort[1])
            smoothed_mean = np.array(
                [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
                 for row in neu_seq_sort[s,l_idx:r_idx]])
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
            # rearrange the matrix.
            mean = neu_seq[s,:][sort_idx_neu,:].copy()
            # plot heatmap.
            heatmap_exc = apply_colormap(mean[labels[s]==-1,:], cmap_exc)
            heatmap_inh = apply_colormap(mean[labels[s]==1,:], cmap_inh)
            neu_h = np.concatenate([heatmap_exc, heatmap_inh], axis=0)
            ax.imshow(neu_h, interpolation='nearest', aspect='auto')
            adjust_layout_heatmap(ax)
            ax.set_ylabel('neuron id (sorted)')
            ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
            ax.set_xticks([0, zero, len(neu_time)])
            ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
            ax.set_yticks([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
            ax.set_yticklabels([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
            # add coloarbar.
            if colorbar:
                if heatmap_exc.shape[0] != 0:
                    cbar_exc = ax.figure.colorbar(
                        plt.cm.ScalarMappable(cmap=cmap_exc), ax=ax, ticks=[0.2,0.8], aspect=100)
                    cbar_exc.ax.set_ylabel('excitory', rotation=-90, va="bottom")
                    cbar_exc.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
                if heatmap_inh.shape[0] != 0:
                    cbar_inh = ax.figure.colorbar(
                        plt.cm.ScalarMappable(cmap=cmap_inh), ax=ax, ticks=[0.2,0.8], aspect=100)
                    cbar_inh.ax.set_ylabel('inhibitory', rotation=-90, va="bottom")
                    cbar_inh.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)

    def plot_win_mag_box(self, ax, neu_seq, neu_time, win_base, color, c_time, offset):
        win_early = [0,250]
        win_late  = [250,500]
        mean_base = get_base_mean_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]), neu_time, c_time, win_base)
        [mean_early, sem_early] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_early[0], win_early[1])
        [mean_late, sem_late] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_late[0], win_late[1])
        mean_early -= mean_base
        mean_late -= mean_base
        ax.errorbar(
            0 + offset,
            mean_early, sem_early,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.errorbar(
            1 + offset,
            mean_late, sem_late,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(
            'response magnitude df/f (mean$\pm$sem) \n baseline [{},{}] ms'.format(
                win_base[0], win_base[1]))
        ax.set_xticks([0,1])
        ax.set_xticklabels(['early \n [{},{}] ms'.format(win_early[0], win_early[1]),
                            'late \n [{},{}] ms'.format(win_late[0], win_late[1])])
        ax.set_xlim([-0.5, 2.5])

    def plot_heatmap_trials(self, ax, neu_seq, neu_time, cmap, norm=True):
        if not np.isnan(np.sum(neu_seq)) and len(neu_seq)>0:
            if len(neu_seq.shape) == 3:
                mean = np.mean(neu_seq, axis=1)
            else:
                mean = neu_seq
            if norm:
                for i in range(mean.shape[0]):
                    mean[i,:] = norm01(mean[i,:])
            zero = np.searchsorted(neu_time, 0)
            img = ax.imshow(
                mean, interpolation='nearest', aspect='auto', cmap=cmap)
            adjust_layout_heatmap(ax)
            ax.set_ylabel('trial id')
            ax.axvline(zero, color='black', lw=1, linestyle='--')
            ax.set_xticks([0, zero, len(neu_time)])
            ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
            ax.set_yticks([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
            ax.set_yticklabels([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
            cbar = ax.figure.colorbar(img, ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
            cbar.ax.set_yticklabels(['0.2', '0.8'])
