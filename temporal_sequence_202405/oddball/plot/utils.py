#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap


# normalization into [0,1].
def norm01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)


# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# mark stim after omi as outlier.
def exclude_post_odd_stim(stim_labels):
    stim_labels_mark = stim_labels.copy()
    idx_post = np.diff(stim_labels_mark[:,2]==-1, prepend=0)
    idx_post[idx_post==1] = 0
    idx_post[idx_post==-1] = 1
    idx_post = idx_post.astype('bool')
    stim_labels_mark[idx_post,2] = -1
    return stim_labels_mark


# compute mean and sem for 3d array neu_seq
def get_mean_sem(neu_seq):
    m = np.mean(neu_seq.reshape(-1, neu_seq.shape[-1]), axis=0)
    s = sem(neu_seq.reshape(-1, neu_seq.shape[-1]), axis=0)
    return m, s


# compute baseline within given time window.
def get_base_mean_win(neu_seq, neu_time, c_time, win_base):
    pct = 30
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, win_base[0], win_base[1])
    neu = neu_seq[:, l_idx:r_idx].copy().reshape(-1)
    neu = neu[neu<np.percentile(neu, pct)]
    mean_base = np.mean(neu)
    return mean_base

    
# compute mean and sem across trials for mean df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time):
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win_mean = np.mean(neu_seq[:, l_idx:r_idx], axis=1)
    neu_mean = np.mean(neu_win_mean)
    neu_sem = sem(neu_win_mean)
    return neu_mean, neu_sem


# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.argmin(np.abs(timestamps-(c_time+l_time)))
    r_idx = np.argmin(np.abs(timestamps-(c_time+r_time)))
    return l_idx, r_idx


# get subsequence index with given start and end.
def get_sub_time_idx(time, start, end):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx


# find index for each epoch.
def get_epoch_idx(stim_labels):
    n = len(stim_labels[:,3])
    len_epoch = n // 4
    epoch_idx = [np.zeros(n, dtype=bool) for _ in range(4)]
    for i in range(4):
        epoch_idx[i][i*len_epoch:(i+1)*len_epoch] = True
    return epoch_idx


# find expected isi.
def get_expect_time(stim_labels):
    idx_normal = stim_labels[:,2]>1
    expect = stim_labels[1:,0] - stim_labels[:-1,1]
    expect = np.median(expect[idx_normal[:-1]])
    return expect


# get grating index for short/long or pre/post
def get_odd_stim_prepost_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,5]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,5]==1)
    idx_post_short = np.zeros_like(idx_pre_short)
    idx_post_short[1:] = idx_pre_short[:-1]
    idx_post_long = np.zeros_like(idx_pre_long)
    idx_post_long[1:] = idx_pre_long[:-1]
    return idx_pre_short, idx_pre_long, idx_post_short, idx_post_long


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


# normalize and apply colormap
def apply_colormap(data, cmap):
    if data.shape[1] == 0:
        return np.zeros((0, data.shape[1], 3))
    for i in range(data.shape[0]):
        data[i,:] = norm01(data[i,:])
    data_heatmap = cmap(data)
    data_heatmap = data_heatmap[..., :3]
    return data_heatmap


# adjust layout for align trajectory.
def adjust_layout_js(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('joystick deflection (deg)')
    ax.legend(loc='upper right')

    
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
    
    def __init__(self, labels):
        self.labels = labels
        
    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])
    
    def plot_vol(self, ax, st, sv, c, u, l):
        v = np.mean(sv, axis=0)
        v = rescale(v, u, l)
        ax.plot(st, v, color=c, lw=0.5, linestyle=':')

    def plot_win_mag_box(self, ax, neu_seq, neu_time, win_base, color, c_time, offset):
        win_early = [0,250]
        win_late  = [250,500]
        mean_base = get_base_mean_win(
            neu_seq.reshape(-1, neu_seq.shape[2]), neu_time, c_time, win_base)
        [mean_early, sem_early] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[2]),
            neu_time, c_time, win_early[0], win_early[1])
        [mean_late, sem_late] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[2]),
            neu_time, c_time, win_late[0], win_late[1])
        mean_early -= mean_base
        mean_late -= mean_base
        ax.errorbar(
            0 + offset,
            mean_early, sem_early,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=1)
        ax.errorbar(
            1 + offset,
            mean_late, sem_late,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=1)
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
        
    def plot_heatmap_neuron(self, ax, neu_seq, neu_time, neu_seq_sort):
        if len(neu_seq) > 0:
            _, _, _, cmap_exc = get_roi_label_color([-1], 0)
            _, _, _, cmap_inh = get_roi_label_color([1], 0)
            zero = np.argmin(np.abs(neu_time - 0))
            smoothed_mean = np.array(
                [np.convolve(row, np.ones(5)/5, mode='same')
                 for row in np.mean(neu_seq_sort, axis=0)])
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
            mean = np.mean(neu_seq, axis=0)
            mean = mean[sort_idx_neu,:].copy()
            heatmap_exc = apply_colormap(mean[self.labels==-1,:], cmap_exc)
            heatmap_inh = apply_colormap(mean[self.labels==1,:], cmap_inh)
            neu_h = np.concatenate([heatmap_exc, heatmap_inh], axis=0)
            ax.imshow(neu_h, interpolation='nearest', aspect='auto')
            adjust_layout_heatmap(ax)
            ax.set_ylabel('neuron id (sorted)')
            ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
            ax.set_xticks([0, zero, len(neu_time)])
            ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
            ax.set_yticks([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
            ax.set_yticklabels([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
            if heatmap_exc.shape[0] != 0:
                cbar_exc = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_exc), ax=ax, ticks=[0.2,0.8], aspect=100)
                cbar_exc.ax.set_ylabel('excitory', rotation=-90, va="bottom")
                cbar_exc.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
            if heatmap_inh.shape[0] != 0:
                cbar_inh = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_inh), ax=ax, ticks=[0.2,0.8], aspect=100)
                cbar_inh.ax.set_ylabel('inhibitory', rotation=-90, va="bottom")
                cbar_inh.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)

    def plot_heatmap_trials(self, ax, neu_seq, neu_time, cmap, norm=True):
        if len(neu_seq.shape) == 3:
            mean = np.mean(neu_seq, axis=1)
        else:
            mean = neu_seq
        if norm:
            for i in range(mean.shape[0]):
                mean[i,:] = norm01(mean[i,:])
        zero = np.argmin(np.abs(neu_time - 0))
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