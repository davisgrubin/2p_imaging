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


# cut sequence into the same length as the shortest one given pivots.
def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data


# align coolected sequences.
def align_neu_seq_utils(neu_seq, neu_time):
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    return neu_seq, neu_time


# extract response around stimulus.
def get_stim_response(
        neural_trials,
        l_frames, r_frames
        ):
    stim_labels = neural_trials['stim_labels']
    dff = neural_trials['dff']
    time = neural_trials['time']
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq  = []
    pre_isi = []
    # loop over stimulus.
    for stim_id in range(1, stim_labels.shape[0]-1):
        idx = np.argmin(np.abs(time - stim_labels[stim_id,0]))
        if idx > l_frames and idx < len(time)-r_frames:
            # signal response.
            f = dff[:, idx-l_frames : idx+r_frames]
            f = np.expand_dims(f, axis=0)
            neu_seq.append(f)
            # signal time stamps.
            t = time[idx-l_frames : idx+r_frames] - time[idx]
            neu_time.append(t)
            # stimulus.
            stim_seq.append(np.array(
                [[stim_labels[stim_id-1,0]-stim_labels[stim_id,0],
                 stim_labels[stim_id-1,1]-stim_labels[stim_id,0]],
                 [0,
                  stim_labels[stim_id,1]-stim_labels[stim_id,0]],
                 [stim_labels[stim_id+1,0]-stim_labels[stim_id,0],
                  stim_labels[stim_id+1,1]-stim_labels[stim_id,0]]]
                ).reshape(1,3,2))
            # isi before oddball
            pre_isi.append(np.array([stim_labels[stim_id,0]-stim_labels[stim_id-1,1]]))
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = np.concatenate(stim_seq, axis=0)
    pre_isi = np.concatenate(pre_isi, axis=0)
    return [neu_seq, neu_time, stim_seq, pre_isi]


# compute omission timestamp as expected stim.
def get_expect_stim_time(stim_labels):
    idx_short = (stim_labels[:,2]>1) * (stim_labels[:,3]==0)
    idx_long  = (stim_labels[:,2]>1) * (stim_labels[:,3]==1)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    img_dur = np.mean(stim_labels[:,1] - stim_labels[:,0])
    expect_short = img_dur + np.median(isi[idx_short[:-1]])
    expect_long  = img_dur + np.median(isi[idx_long[:-1]])
    return expect_short, expect_long


# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.argmin(np.abs(timestamps-(c_time+l_time)))
    r_idx = np.argmin(np.abs(timestamps-(c_time+r_time)))
    return l_idx, r_idx


# compute mean and sem across trials for mean df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time):
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win_mean = np.mean(neu_seq[:, l_idx:r_idx], axis=1)
    neu_mean = np.mean(neu_win_mean)
    neu_sem = sem(neu_win_mean)
    return neu_mean, neu_sem


# bin the data with timestamps.
def get_bin_stat(data, time, time_range, bin_size):
    bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(time, bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = data[bin_indices == i]
        m = np.mean(bin_values) if len(bin_values) > 0 else np.nan
        s = sem(bin_values) if len(bin_values) > 0 else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


# find index for each epoch.
def get_epoch_idx(stim_labels):
    n = len(stim_labels[:,3])
    len_block = n // 4
    len_epoch = len_block // 2
    epoch_short = [np.zeros(n, dtype=bool) for _ in range(4)]
    epoch_long  = [np.zeros(n, dtype=bool) for _ in range(4)]
    short_blocks = np.where(np.array(stim_labels[:,3]) == 0)[0]
    long_blocks = np.where(np.array(stim_labels[:,3]) == 1)[0]
    epoch_short[0][short_blocks[:len_epoch]] = True
    epoch_short[1][short_blocks[len_epoch:len_block]] = True
    epoch_short[2][short_blocks[len_block:len_block + len_epoch]] = True
    epoch_short[3][short_blocks[len_block + len_epoch:]] = True
    epoch_long[0][long_blocks[:len_epoch]] = True
    epoch_long[1][long_blocks[len_epoch:len_block]] = True
    epoch_long[2][long_blocks[len_block:len_block + len_epoch]] = True
    epoch_long[3][long_blocks[len_block + len_epoch:]] = True
    epoch_short = [e[1:-1] for e in epoch_short]
    epoch_long = [e[1:-1] for e in epoch_long]
    return epoch_short, epoch_long


# get response for each epoch.
def get_epoch_response(neu_cate, stim_seq, stim_labels, epoch, idx):
    stim_seq_epoch = []
    for i in range(4):
        idx_epoch = epoch[i] * idx
        stim_seq_epoch.append(np.mean(stim_seq[idx_epoch,:,:], axis=0))
    expect_epoch = []
    for i in range(4):
        stim_labels_epoch = stim_labels[epoch[i],:].copy()
        isi = stim_labels_epoch[1:,0] - stim_labels_epoch[:-1,1]
        img_dur = np.mean(stim_labels_epoch[:,1] - stim_labels_epoch[:,0])
        expect_epoch.append(img_dur + np.median(isi))
    neu_epoch = []
    for i in range(4):
        idx_epoch = epoch[i] * idx
        neu_epoch.append(neu_cate[idx_epoch,:,:])
    return [stim_seq_epoch, expect_epoch, neu_epoch]


# get grating index for short/long or pre/post
def get_omi_stim_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,3]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,3]==1)
    idx_post = np.diff(stim_labels[:,2]==-1, prepend=0)
    idx_post[idx_post==1] = 0
    idx_post[idx_post==-1] = 1
    idx_post = idx_post.astype('bool')
    idx_post_short = idx_post * (stim_labels[:,3]==0)
    idx_post_long  = idx_post * (stim_labels[:,3]==1)
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


# adjust layout for grand average.
def adjust_layout_grand(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper left')
    

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])


class plotter_utils:
    
    def __init__(
            self,
            neural_trials, labels
            ):
        self.l_frames = 150
        self.r_frames = 150
        self.cut_frames = 50
        [self.epoch_short, self.epoch_long] = get_epoch_idx(
            neural_trials['stim_labels'])
        self.stim_labels = neural_trials['stim_labels'][1:-1,:]
        self.labels = labels
        [self.neu_seq, self.neu_time, self.stim_seq, self.pre_isi] = get_stim_response(
            neural_trials, self.l_frames, self.r_frames)
        self.expect_short, self.expect_long = get_expect_stim_time(
            self.stim_labels)
        [self.idx_pre_short, self.idx_pre_long,
         self.idx_post_short, self.idx_post_long] = get_omi_stim_idx(
            self.stim_labels)

    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])
    
    def plot_win_mean_sem_box(self, ax, neu_seq_list, colors, lbl, c_time_list, offset):
        win_base  = [-100,0]
        win_early = [0,250]
        win_late  = [250,500]
        win = [win_base, win_early, win_late]
        pos = [-0.15, 0, 0.15]
        center = np.arange(len(lbl))+1
        for i in range(len(lbl)):
            for j in range(3):
                [neu_mean, neu_sem] = get_mean_sem_win(
                    neu_seq_list[i].reshape(-1, self.l_frames+self.r_frames),
                    self.neu_time, c_time_list[i], win[j][0], win[j][1])
                ax.errorbar(
                    center[i] + pos[j] + offset,
                    neu_mean, neu_sem,
                    color=colors[j],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.set_xticks(center)
        ax.set_xticklabels(lbl)
        ax.set_xlim([0.5,len(lbl)+1.5])
    
    def plot_heatmap_trials(self, ax, neu_seq, cmap, norm=True):
        if len(neu_seq.shape) == 3:
            mean = np.mean(neu_seq, axis=1)
        else:
            mean = neu_seq
        if norm:
            for i in range(mean.shape[0]):
                mean[i,:] = norm01(mean[i,:])
        zero = np.argmin(np.abs(self.neu_time - 0))
        img = ax.imshow(
            mean, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
        ax.set_xticks([0, zero, len(self.neu_time)])
        ax.set_xticklabels([int(self.neu_time[0]), 0, int(self.neu_time[-1])])
        ax.set_yticks([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
        ax.set_yticklabels([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
        cbar = ax.figure.colorbar(img, ax=ax, ticks=[0.2,0.8], aspect=100)
        cbar.ax.set_ylabel('response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])

    def plot_heatmap_neuron(self, ax, neu_seq):
        _, _, _, cmap_exc = get_roi_label_color([-1], 0)
        _, _, _, cmap_inh = get_roi_label_color([1], 0)
        mean = np.mean(neu_seq, axis=0)
        zero = np.argmin(np.abs(self.neu_time - 0))
        smoothed_mean = np.array([np.convolve(row, np.ones(5)/5, mode='same') for row in mean])
        sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
        mean = mean[sort_idx_neu,:].copy()
        heatmap_exc = apply_colormap(mean[self.labels==-1,:], cmap_exc)
        heatmap_inh = apply_colormap(mean[self.labels==1,:], cmap_inh)
        neu_h = np.concatenate([heatmap_exc, heatmap_inh], axis=0)
        ax.imshow(neu_h, interpolation='nearest', aspect='auto')
        adjust_layout_heatmap(ax)
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('neuron id (sorted)')
        ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
        ax.set_xticks([0, zero, len(self.neu_time)])
        ax.set_xticklabels([int(self.neu_time[0]), 0, int(self.neu_time[-1])])
        ax.set_yticks([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
        ax.set_yticklabels([0, int(neu_h.shape[0]/3), int(neu_h.shape[0]*2/3), int(neu_h.shape[0])])
        if heatmap_exc.shape[0] != 0:
            cbar_exc = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_exc),
                                          ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar_exc.ax.set_ylabel('excitory', rotation=-90, va="bottom")
            cbar_exc.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
        if heatmap_inh.shape[0] != 0:
            cbar_inh = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_inh),
                                          ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar_inh.ax.set_ylabel('inhibitory', rotation=-90, va="bottom")
            cbar_inh.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
        
    def plot_omi_normal_pre(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        stim_seq_short = np.mean(self.stim_seq[self.idx_pre_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[self.idx_pre_long,:,:], axis=0)
        neu_short = neu_cate[self.idx_pre_short,:,:]
        neu_long  = neu_cate[self.idx_pre_long,:,:]
        mean_short = np.mean(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        mean_long  = np.mean(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_short  = sem(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_long   = sem(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='pre')
        ax.axvline(
            self.expect_short,
            color=color1, label='omi (short)',
            lw=1, linestyle='--')
        ax.axvline(
            self.expect_long,
            color=color2, label='omi (long)',
            lw=1, linestyle='--')
        ax.fill_between(
            stim_seq_short[2,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='post (short)')
        ax.fill_between(
            stim_seq_long[2,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='post (long)')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_grand(ax)
        ax.set_xlim([self.neu_time[np.argmin(np.abs(self.neu_time))-self.cut_frames],
                     self.neu_time[-1]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre omi stim (ms)')
    
    def plot_omi_normal_post(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        stim_seq_short = np.mean(self.stim_seq[self.idx_post_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[self.idx_post_long,:,:], axis=0)
        neu_short = neu_cate[self.idx_post_short,:,:]
        neu_long  = neu_cate[self.idx_post_long,:,:]
        mean_short = np.mean(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        mean_long  = np.mean(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_short  = sem(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_long   = sem(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.axvline(
            -self.expect_short+stim_seq_short[1,1],
            color=color1, label='omi (short)',
            lw=1, linestyle='--')
        ax.axvline(
            -self.expect_long+stim_seq_short[1,1],
            color=color2, label='omi (long)',
            lw=1, linestyle='--')
        ax.fill_between(
            stim_seq_short[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre (short)')
        ax.fill_between(
            stim_seq_long[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='pre (long)')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_grand(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')
    
    def plot_omi_normal_shortlong(self, ax, idx_pre, idx_post, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        stim_seq_pre = np.mean(self.stim_seq[idx_pre,:,:], axis=0)
        neu_pre  = neu_cate[idx_pre,:,:]
        neu_post = neu_cate[idx_post,:,:]
        mean_pre  = np.mean(neu_pre.reshape(-1, self.l_frames+self.r_frames), axis=0)
        mean_post = np.mean(neu_post.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_pre  = sem(neu_pre.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_post = sem(neu_post.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([mean_pre, mean_post]) + np.max([sem_pre, sem_post])
        lower = np.min([mean_pre, mean_post]) - np.max([sem_pre, sem_post])
        ax.fill_between(
            stim_seq_pre[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_pre, sem_pre, color1, 'pre')
        self.plot_mean_sem(ax, self.neu_time, mean_post, sem_post, color2, 'post')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_omi_normal_box(self, ax, cate=None, roi_id=None):
        lbl = ['pre omi stim','omi','post omi stim']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        idx_short = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==0)
        idx_long  = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==1)
        neu_short = neu_cate[idx_short,:,:]
        neu_long  = neu_cate[idx_long,:,:]
        stim_seq_short = np.mean(self.stim_seq[idx_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[idx_long,:,:], axis=0)
        self.plot_win_mean_sem_box(
            ax, [neu_short]*3, color_short, lbl,
            [0, self.expect_short, stim_seq_short[1,0]], 0.04)
        self.plot_win_mean_sem_box(
            ax, [neu_long]*3, color_long, lbl,
            [0, self.expect_long, stim_seq_long[1,0]], -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')
    
    def plot_omi_post_isi(
            self, ax,
            idx_post, bins, isi_labels, colors,
            cate=None, roi_id=None
            ):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        bin_idx = np.digitize(self.pre_isi, bins)-1
        stim_seq_post = np.mean(self.stim_seq[idx_post,:,:], axis=0)
        m = []
        s = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                bin_idx_post = np.zeros_like(bin_idx, dtype=bool)
                bin_idx_post[1:] = (bin_idx==i)[:-1]
                isi_idx = bin_idx_post*idx_post
                neu = neu_cate[isi_idx,:,:].reshape(-1, self.l_frames+self.r_frames)
                self.plot_mean_sem(
                    ax, self.neu_time, np.mean(neu, axis=0), sem(neu, axis=0),
                    colors[i], isi_labels[i])
                m.append(np.mean(neu, axis=0))
                s.append(sem(neu, axis=0))
        m = np.concatenate(m)
        s = np.concatenate(s)
        m = m[~np.isnan(m)]
        s = s[~np.isnan(s)]
        upper = np.max(m) + np.max(s)
        lower = np.min(m) - np.max(s)
        ax.fill_between(
            stim_seq_post[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='grey', alpha=0.15, step='mid', label='post')
        adjust_layout_grand(ax)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    def plot_omi_post_isi_box(self, ax, idx_post, bins, isi_labels, cate=None, roi_id=None):
        colors = ['black', 'grey', 'silver']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        bin_idx = np.digitize(self.pre_isi, bins)-1
        neu = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                bin_idx_post = np.zeros_like(bin_idx, dtype=bool)
                bin_idx_post[1:] = (bin_idx==i)[:-1]
                isi_idx = bin_idx_post*idx_post
                neu.append(neu_cate[isi_idx,:,:].reshape(
                    -1, self.l_frames+self.r_frames))
        self.plot_win_mean_sem_box(ax, neu, colors, isi_labels, [0]*4, 0.04)
                
    def plot_omi_isi_box(self, ax, idx, isi, c_time, offset, lbl, cate=None, roi_id=None):
        l_time = 0
        r_time = 300
        time_range = [200,2000]
        bin_size = 200
        l_idx, r_idx = get_frame_idx_from_time(
            self.neu_time, c_time, l_time, r_time)
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        neu_cate = neu_cate[idx,:,:]
        mag = np.mean(neu_cate[:,:, l_idx:r_idx], axis=2).reshape(-1)
        bins, bin_mean, bin_sem = get_bin_stat(mag, isi, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
            bin_mean,
            bin_sem,
            color=color, capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1, label=lbl)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('preceeding isi (ms)')
        ax.set_ylabel('df/f at omission with {}ms window (mean$\pm$sem)'.format(
            r_time-l_time))
        ax.legend(loc='upper right')
        
    def plot_omi_context(self, ax, img_id, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        n_s = neu_cate[self.idx_post_short*(self.stim_labels[:,2]==img_id),:,:].reshape(
            -1, self.l_frames+self.r_frames)
        n_l = neu_cate[self.idx_post_long*(self.stim_labels[:,2]==img_id),:,:].reshape(
            -1, self.l_frames+self.r_frames)
        mean_short = np.mean(n_s, axis=0)
        mean_long  = np.mean(n_l, axis=0)
        sem_short  = sem(n_s, axis=0)
        sem_long   = sem(n_l, axis=0)
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        stim_seq_short = np.mean(self.stim_seq[self.idx_post_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[self.idx_post_long,:,:], axis=0)
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.axvline(
            -self.expect_short+stim_seq_short[1,1],
            color=color1, label='omi (short)',
            lw=1, linestyle='--')
        ax.axvline(
            -self.expect_long+stim_seq_short[1,1],
            color=color2, label='omi (long)',
            lw=1, linestyle='--')
        ax.fill_between(
            stim_seq_short[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre (short)')
        ax.fill_between(
            stim_seq_long[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='pre (long)')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_grand(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')

    def plot_omi_context_box(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        neu_short = []
        neu_long = []
        for img_id in [2,3,4,5]:
            n_s = neu_cate[self.idx_post_short*(self.stim_labels[:,2]==img_id),:,:].reshape(
                -1, self.l_frames+self.r_frames)
            n_l = neu_cate[self.idx_post_long*(self.stim_labels[:,2]==img_id),:,:].reshape(
                -1, self.l_frames+self.r_frames)
            neu_short.append(n_s)
            neu_long.append(n_l)
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, [0]*4, 0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, [0]*4, -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')

    def plot_omi_epoch_post(self, ax, epoch, idx, colors, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        [stim_seq_epoch, expect_epoch, neu_epoch] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels, epoch, idx)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            neu_mean.append(np.mean(neu_epoch[i].reshape(
                -1, self.l_frames+self.r_frames), axis=0))
            neu_sem.append(sem(neu_epoch[i].reshape(
                -1, self.l_frames+self.r_frames), axis=0))
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            stim_seq_epoch[0][1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid')
        for i in range(4):
            ax.axvline(
                -expect_epoch[i]+stim_seq_epoch[i][1,1],
                color=colors[i],
                lw=1, linestyle='--')
            ax.fill_between(
                stim_seq_epoch[i][0,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], alpha=0.15, step='mid')
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        ax.plot([], lw=1, color='grey', linestyle='--', label='omi')
        adjust_layout_grand(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')
    
    def plot_omi_epoch_post_box(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        [_, _, neu_short] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels,
            self.epoch_short, self.idx_post_short)
        [_, _, neu_long] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels,
            self.epoch_long, self.idx_post_long)
        neu_short = [n_s.reshape(-1, self.l_frames+self.r_frames) for n_s in neu_short]
        neu_long = [n_l.reshape(-1, self.l_frames+self.r_frames) for n_l in neu_long]
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, [0]*4, 0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, [0]*4, -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')

    # roi mean omission response comparison between pre and post for short. 
    def roi_omi_normal_short(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # roi mean omission response comparison between pre and post for long. 
    def roi_omi_normal_long(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # roi mean response quantification.
    def roi_omi_normal_box(self, ax, roi_id):
        self.plot_omi_normal_box(ax, roi_id=roi_id)
        ax.set_title('mean response comparison')
    
    # roi omission response and isi before omission for short.
    def roi_omi_isi_short(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_short], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi, self.expect_short, 0, roi_id=roi_id)
        ax.set_title('omission response and preceeding isi (short)')
    
    # roi omission response and isi before omission for long.
    def roi_omi_isi_long(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_long], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi, self.expect_long, 0, roi_id=roi_id)
        ax.set_title('omission response and preceeding isi (long)')
    
    # roi post omission stimulus response and isi before omission for short.
    def roi_omi_isi_post_short(self, ax, roi_id):
        idx = np.diff(self.idx_post_short, append=0)
        idx[idx==-1] = 0
        idx = idx.astype('bool')
        isi = np.tile(self.pre_isi[idx], 1)
        self.plot_omi_isi_box(ax, self.idx_post_short, isi, 0, 0, roi_id=roi_id)
        ax.set_title('post omi stim response and preceeding isi (short)')
    
    # roi post omission stimulus response and isi before omission for long.
    def roi_omi_isi_post_long(self, ax, roi_id):
        idx = np.diff(self.idx_post_long, append=0)
        idx[idx==-1] = 0
        idx = idx.astype('bool')
        isi = np.tile(self.pre_isi[idx], 1)
        self.plot_omi_isi_box(ax, self.idx_post_long, isi, 0, 0, roi_id=roi_id)
        ax.set_title('post omi stim response and preceeding isi (long)')
    
    # mean response to short long context.
    def roi_omi_context(self, axs, roi_id):
        titles = [
            'post omi stim response around img#1',
            'post omi stim response around img#2',
            'post omi stim response around img#3',
            'post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, roi_id=roi_id)
            axs[i].set_title(titles[i])
    
    # roi mean response to short long context quantification.
    def roi_omi_context_box(self, ax, roi_id):
        self.plot_omi_context_box(ax, roi_id=roi_id)
        ax.set_title('post omi stim response around different image')
        
        
class plotter_VIPTD_G8_align_omi(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        
    # excitory mean response to pre omission stimulus.
    def omi_normal_pre_exc(self, ax):
        self.plot_omi_normal_pre(ax, cate=-1)
        ax.set_title(
            'exitory mean response to pre omi stim')
    
    # inhibitory mean response to pre omission stimulus.
    def omi_normal_pre_inh(self, ax):
        self.plot_omi_normal_pre(ax, cate=1)
        ax.set_title(
            'inhibitory mean response to pre omi stim')
    
    # mean response to pre omission stimulus average across trial for short.
    def omi_normal_pre_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (short)')
    
    # mean response to pre omission stimulus average across trial for long.
    def omi_normal_pre_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (long)')
    
    # excitory mean response to post omission stimulus.
    def omi_normal_post_exc(self, ax):
        self.plot_omi_normal_post(ax, cate=-1)
        ax.set_title(
            'excitory mean response to post omi stim')
    
    # inhibitory mean response to post omission stimulus.
    def omi_normal_post_inh(self, ax):
        self.plot_omi_normal_post(ax, cate=1)
        ax.set_title(
            'inhibitory mean response to post omi stim')
    
    # mean response to post omission stimulus average across trial for short.
    def omi_normal_post_short_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (short)')
    
    # mean response to post omission stimulus average across trial for long.
    def omi_normal_post_long_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (long)')
        
    # excitory mean omission response comparison between pre and post for short. 
    def omi_normal_exc_short(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=-1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # excitory mean omission response comparison between pre and post for long. 
    def omi_normal_exc_long(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=-1)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # inhibitory mean omission response comparison between pre and post for short. 
    def omi_normal_inh_short(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # inhibitory mean omission response comparison between pre and post for long. 
    def omi_normal_inh_long(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=1)
        ax.set_title('mean stim response around omi comparison (long)')
        
    # excitory mean response quantification.
    def omi_normal_exc_box(self, ax):
        self.plot_omi_normal_box(ax, cate=-1)
        ax.set_title('excitory mean response comparison')
    
    # inhibitory mean response quantification.
    def omi_normal_inh_box(self, ax):
        self.plot_omi_normal_box(ax, cate=1)
        ax.set_title('inhibitory mean response comparison')
    
    # excitory mean response to post omi stim across epoch for short.
    def omi_epoch_post_exc_short(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_epoch_post(ax, self.epoch_short, self.idx_post_short, colors, cate=-1)
        ax.set_title(
            'excitory mean response across epoch (short)')
    
    # excitory mean response to post omi stim across epoch for long.
    def omi_epoch_post_exc_long(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_epoch_post(ax, self.epoch_long, self.idx_post_long, colors, cate=-1)
        ax.set_title('excitory mean response across epoch (long)')
    
    # inhibitory mean response to post omi stim across epoch for short.
    def omi_epoch_post_inh_short(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_epoch_post(ax, self.epoch_short, self.idx_post_short, colors, cate=1)
        ax.set_title('inhibitory mean response across epoch (short)')
    
    # inhibitory mean response to post omi stim across epoch for long.
    def omi_epoch_post_inh_long(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_epoch_post(ax, self.epoch_long, self.idx_post_long, colors, cate=1)
        ax.set_title('inhibitory mean response across epoch (long)')
    
    # excitory mean response to post omi stim across epoch quantification.
    def omi_epoch_post_exc_box(self, ax):
        self.plot_omi_epoch_post_box(ax, cate=-1)
        ax.set_title('excitory mean response on post omi stim across epoch')
    
    # inhibitory mean response to post omi stim across epoch quantification.
    def omi_epoch_post_inh_box(self, ax):
        self.plot_omi_epoch_post_box(ax, cate=1)
        ax.set_title('inhibitory mean response on post omi stim across epoch')
    
    # excitory mean response to post omi stim single trial heatmap for short.
    def omi_post_exc_short_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_short,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('excitory single trial response to post omi stim (short)')
    
    # excitory mean response to post omi stim single trial heatmap for short.
    def omi_post_exc_long_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_long,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('excitory single trial response to post omi stim (long)')
        
    # inhibitory mean response to post omi stim single trial heatmap for short.
    def omi_post_inh_short_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.idx_post_short,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('inhibitory single trial response to post omi stim (short)')
    
    # inhibitory mean response to post omi stim single trial heatmap for short.
    def omi_post_inh_long_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.idx_post_long,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('inhibitory single trial response to post omi stim (long)')
        
    # omission response and isi before omission for short.
    def omi_isi_short(self, ax):
        isi_exc = np.tile(self.pre_isi[self.idx_pre_short], np.sum(self.labels==-1))
        isi_inh = np.tile(self.pre_isi[self.idx_pre_short], np.sum(self.labels==1))
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi_exc, self.expect_short, -0.02,
                          'excitory', cate=-1)
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi_inh, self.expect_short, 0.02,
                          'inhibitory', cate=1)
        ax.set_title('omission response and preceeding isi (short)')
        
    # omission response and isi before omission for long.
    def omi_isi_long(self, ax):
        isi_exc = np.tile(self.pre_isi[self.idx_pre_long], np.sum(self.labels==-1))
        isi_inh = np.tile(self.pre_isi[self.idx_pre_long], np.sum(self.labels==1))
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi_exc, self.expect_long, -0.02,
                          'excitory', cate=-1)
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi_inh, self.expect_long, 0.02,
                          'inhibitory', cate=1)
        ax.set_title('omission response and preceeding isi (long)')

    # excitory mean response to short long context.
    def omi_context_exc(self, axs):
        titles = [
            'excitory post omi stim response around img#1',
            'excitory post omi stim response around img#2',
            'excitory post omi stim response around img#3',
            'excitory post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=-1)
            axs[i].set_title(titles[i])
    
    # inhibitory mean response to short long context.
    def omi_context_inh(self, axs):
        titles = [
            'inhibitory post omi stim response around img#1',
            'inhibitory post omi stim response around img#2',
            'inhibitory post omi stim response around img#3',
            'inhibitory post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=1)
            axs[i].set_title(titles[i])
    
    # excitory mean response to short long context quantification.
    def omi_context_exc_box(self, ax):
        self.plot_omi_context_box(ax, cate=-1)
        ax.set_title('excitory post omi stim response around different image')
    
    # inhibitory mean response to short long context quantification.
    def omi_context_inh_box(self, ax):
        self.plot_omi_context_box(ax, cate=1)
        ax.set_title('inhibitory post omi stim response around different image')
    
    # excitory response to post omi stim with proceeding isi for short.
    def omi_post_isi_exc_short(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_short,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (short)')
    
    # excitory response to post omi stim with proceeding isi for short quantification.
    def omi_post_isi_exc_short_box(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        self.plot_omi_post_isi_box(ax, self.idx_post_short, bins, isi_labels, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (short)')
            
    # excitory response to post omi stim with proceeding isi for long.
    def omi_post_isi_exc_long(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_long,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (long)')
    
    # excitory response to post omi stim with proceeding isi for long quantification.
    def omi_post_isi_exc_long_box(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        self.plot_omi_post_isi_box(ax, self.idx_post_long, bins, isi_labels, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (long)')
    
    # inhibitory response to post omi stim with proceeding isi for short.
    def omi_post_isi_inh_short(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_short,
            bins, isi_labels, colors, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (short)')
    
    # inhibitory response to post omi stim with proceeding isi for short quantification.
    def omi_post_isi_inh_short_box(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        self.plot_omi_post_isi_box(ax, self.idx_post_short, bins, isi_labels, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (short)')
            
    # inhibitory response to post omi stim with proceeding isi for long.
    def omi_post_isi_inh_long(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_long,
            bins, isi_labels, colors, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (long)')
    
    # inhibitory response to post omi stim with proceeding isi for long quantification.
    def omi_post_isi_inh_long_box(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        self.plot_omi_post_isi_box(ax, self.idx_post_long, bins, isi_labels, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (long)')
        
class plotter_VIPG8_align_omi(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        self.labels = np.ones_like(labels)
    
    # mean response to pre omission stimulus.
    def omi_normal_pre(self, ax):
        self.plot_omi_normal_pre(ax, cate=1)
        ax.set_title(
            'response to pre omi stim of {} neurons'.format(
            np.sum(self.labels==1)))
    
    # mean response to pre omission stimulus average across trial for short.
    def omi_normal_pre_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (short)')
    
    # mean response to pre omission stimulus average across trial for long.
    def omi_normal_pre_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (long)')
    
    # mean response to post omission stimulus.
    def omi_normal_post(self, ax):
        self.plot_omi_normal_post(ax, cate=1)
        ax.set_title(
            'response to post omi stim of {} neurons'.format(
            np.sum(self.labels==1)))
    
    # mean response to post omission stimulus average across trial for short.
    def omi_normal_post_short_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (short)')
    
    # mean response to post omission stimulus average across trial for long.
    def omi_normal_post_long_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (long)')
    
    # mean omission response comparison between pre and post for short. 
    def omi_normal_short(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # mean omission response comparison between pre and post for long. 
    def omi_normal_long(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=1)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # mean response of quantification.
    def omi_normal_box(self, ax):
        self.plot_omi_normal_box(ax, cate=1)
        ax.set_title(
            'mean response comparison of {} neurons'.format(
            np.sum(self.labels==1)))
    
    # mean response to short long context.
    def omi_context(self, axs):
        titles = [
            'post omi stim response around img#1',
            'post omi stim response around img#2',
            'post omi stim response around img#3',
            'post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=1)
            axs[i].set_title(titles[i])

    # mean response to short long context quantification.
    def omi_context_box(self, ax):
        self.plot_omi_context_box(ax, cate=1)
        ax.set_title('post omi stim response around different image')


class plotter_L7G8_align_omi(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        self.labels = -1*np.ones_like(labels)
    
    # mean response to pre omission stimulus.
    def omi_normal_pre(self, ax):
        self.plot_omi_normal_pre(ax, cate=-1)
        ax.set_title(
            'response to pre omi stim of {} ROIs'.format(
            np.sum(self.labels==-1)))
    
    # mean response to pre omission stimulus average across trial for short.
    def omi_normal_pre_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (short)')
    
    # mean response to pre omission stimulus average across trial for long.
    def omi_normal_pre_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to pre omi stim (long)')
    
    # mean response to post omission stimulus.
    def omi_normal_post(self, ax):
        self.plot_omi_normal_post(ax, cate=-1)
        ax.set_title(
            'response to post omi stim of {} ROIs'.format(
            np.sum(self.labels==1)))
    
    # mean response to post omission stimulus average across trial for short.
    def omi_normal_post_short_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (short)')
    
    # mean response to post omission stimulus average across trial for long.
    def omi_normal_post_long_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (long)')
    
    # mean omission response comparison between pre and post for short. 
    def omi_normal_short(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=-1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # mean omission response comparison between pre and post for long. 
    def omi_normal_long(self, ax):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=-1)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # mean response of quantification.
    def omi_normal_box(self, ax):
        self.plot_omi_normal_box(ax, cate=-1)
        ax.set_title(
            'mean response comparison of {} neurons'.format(
            np.sum(self.labels==1)))
    
    # mean response to short long context.
    def omi_context(self, axs):
        titles = [
            'post omi stim response around img#1',
            'post omi stim response around img#2',
            'post omi stim response around img#3',
            'post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=-1)
            axs[i].set_title(titles[i])

    # mean response to short long context quantification.
    def omi_context_box(self, ax):
        self.plot_omi_context_box(ax, cate=-1)
        ax.set_title('post omi stim response around different image')