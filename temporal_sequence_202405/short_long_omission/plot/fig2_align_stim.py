#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
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
    # loop over stimulus.
    for stim_id in range(stim_labels.shape[0]):
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
                [0, stim_labels[stim_id,1]-stim_labels[stim_id,0]]))
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = np.mean(stim_seq, axis=0)
    return [neu_seq, neu_time, stim_seq]


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


# mark stim after omi as outlier.
def exclude_post_omi_stim(stim_labels):
    stim_labels_mark = stim_labels.copy()
    idx_post = np.diff(stim_labels_mark[:,2]==-1, prepend=0)
    idx_post[idx_post==1] = 0
    idx_post[idx_post==-1] = 1
    idx_post = idx_post.astype('bool')
    stim_labels_mark[idx_post,2] = -1
    return stim_labels_mark


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
    return epoch_short, epoch_long


# normalize and apply colormap
def apply_colormap(data, cmap):
    if data.shape[1] == 0:
        return np.zeros((0, data.shape[1], 3))
    for i in range(data.shape[0]):
        data[i,:] = norm01(data[i,:])
    data_heatmap = cmap(data)
    data_heatmap = data_heatmap[..., :3]
    return data_heatmap


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


# adjust layout for grand average.
def adjust_layout_grand(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper right')
    

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
        self.l_frames = 30
        self.r_frames = 50
        self.stim_labels = exclude_post_omi_stim(neural_trials['stim_labels'])
        self.labels = labels
        [self.epoch_short, self.epoch_long] = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time, self.stim_seq] = get_stim_response(
                neural_trials, self.l_frames, self.r_frames)

    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])
    
    def plot_win_mean_sem_box(self, ax, neu_seq_list, colors, lbl, c_time, offset):
        win_base  = [-100,0]
        win_early = [0,250]
        win_late  = [250,500]
        win = [win_base, win_early, win_late]
        pos = [-0.15, 0, 0.15]
        center = np.arange(len(lbl))+1
        for i in range(len(lbl)):
            [base_mean, _] = get_mean_sem_win(
                neu_seq_list[i].reshape(-1, self.l_frames+self.r_frames),
                self.neu_time, c_time, win_base[0], win_base[1])
            for j in range(3):
                [neu_mean, neu_sem] = get_mean_sem_win(
                    neu_seq_list[i].reshape(-1, self.l_frames+self.r_frames),
                    self.neu_time, c_time, win[j][0], win[j][1])
                if j > 0:
                    neu_mean -= base_mean
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
            cbar_exc = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_exc), ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar_exc.ax.set_ylabel('excitory', rotation=-90, va="bottom")
            cbar_exc.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
        if heatmap_inh.shape[0] != 0:
            cbar_inh = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap_inh), ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar_inh.ax.set_ylabel('inhibitory', rotation=-90, va="bottom")
            cbar_inh.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
    
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
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
        ax.set_xticks([0, zero, len(self.neu_time)])
        ax.set_xticklabels([int(self.neu_time[0]), 0, int(self.neu_time[-1])])
        ax.set_yticks([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
        ax.set_yticklabels([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
        cbar = ax.figure.colorbar(img, ax=ax, ticks=[0.2,0.8], aspect=100)
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        
    def plot_normal(self, ax, cate=None, roi_id=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral', 'grey']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in [2,3,4,5]:
            neu_mean.append(
                np.mean(neu_cate[self.stim_labels[:,2]==i,:,:].reshape(
                    -1, self.l_frames+self.r_frames), axis=0))
            neu_sem.append(
                sem(neu_cate[self.stim_labels[:,2]==i,:,:].reshape(
                    -1, self.l_frames+self.r_frames), axis=0))
        neu_mean.append(np.mean(neu_cate[self.stim_labels[:,2]>0,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0))
        neu_sem.append(sem(neu_cate[self.stim_labels[:,2]>0,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0))
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='silver', alpha=0.25, step='mid', label='stim')
        for i in range(5):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_box(self, ax, cate=None, roi_id=None):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        colors = ['black', 'grey', 'silver']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in [2,3,4,5]:
            neu.append(neu_cate[self.stim_labels[:,2]==i,:,:])
        neu.append(neu_cate[self.stim_labels[:,2]>0,:,:])
        self.plot_win_mean_sem_box(ax, neu, colors, lbl, 0, 0)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=colors[i], label=stages[i])
        ax.legend(loc='upper right')
    
    def plot_normal_epoch(self, ax, epoch, colors, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            idx = epoch[i] * self.stim_labels[:,2]>0
            neu_mean.append(np.mean(neu_cate[idx,:,:].reshape(
                -1, self.l_frames+self.r_frames), axis=0))
            neu_sem.append(sem(neu_cate[idx,:,:].reshape(
                -1, self.l_frames+self.r_frames), axis=0))
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='silver', alpha=0.25, step='mid', label='stim')
        for i in range(4):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_change(self, ax, normal, cate=None, roi_id=None):
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = (idx_pre * (self.stim_labels[:,3]==normal)).astype('bool')
        idx_post = (self.stim_labels[:,2]<-1) * (self.stim_labels[:,3]==normal)
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        neu_mean_pre = np.mean(neu_cate[idx_pre,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        neu_sem_pre = sem(neu_cate[idx_pre,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        neu_mean_post = np.mean(neu_cate[idx_post,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        neu_sem_post = sem(neu_cate[idx_post,:,:].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([neu_mean_pre, neu_mean_post]) + np.max([neu_sem_pre, neu_sem_post])
        lower = np.min([neu_mean_pre, neu_mean_post]) - np.max([neu_sem_pre, neu_sem_post])
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='silver', alpha=0.25, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, neu_mean_pre, neu_sem_pre, color1, 'pre')
        self.plot_mean_sem(ax, self.neu_time, neu_mean_post, neu_sem_post, color2, 'post')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_change_box(self, ax, cate=None, roi_id=None):
        lbl = ['pre', 'post']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre_short = (idx_pre * (self.stim_labels[:,3]==0)).astype('bool')
        idx_post_short = (self.stim_labels[:,2]<-1) * (self.stim_labels[:,3]==0)
        idx_pre_long = (idx_pre * (self.stim_labels[:,3]==1)).astype('bool')
        idx_post_long = (self.stim_labels[:,2]<-1) * (self.stim_labels[:,3]==1)
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_short = [neu_cate[idx_pre_short,:,:].reshape(-1, self.l_frames+self.r_frames),
                     neu_cate[idx_post_short,:,:].reshape(-1, self.l_frames+self.r_frames)]
        neu_long = [neu_cate[idx_pre_long,:,:].reshape(-1, self.l_frames+self.r_frames),
                    neu_cate[idx_post_long,:,:].reshape(-1, self.l_frames+self.r_frames)]    
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, 0, -0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, 0, 0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')
    
    def plot_context(self, ax, img_id, cate=None, roi_id=None):
        colors = ['dodgerblue', 'mediumseagreen', 'hotpink', 'coral', 'grey']
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        if img_id == 5:
            idx_short = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
            idx_long  = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
            neu_short = neu_cate[idx_short,:,:]
            neu_long  = neu_cate[idx_long,:,:]
        else:
            idx_short = (self.stim_labels[:,2]==(img_id+1)) * (self.stim_labels[:,3]==0)
            idx_long  = (self.stim_labels[:,2]==(img_id+1)) * (self.stim_labels[:,3]==1)
            neu_short = neu_cate[idx_short,:,:]
            neu_long  = neu_cate[idx_long,:,:]
        mean_short = np.mean(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        mean_long  = np.mean(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_short  = sem(neu_short.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_long   = sem(neu_long.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=colors[img_id-1], alpha=0.15, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since img change (ms)')
        
    def plot_context_box(self, ax, cate=None, roi_id=None):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_short = []
        neu_long = []
        for i in [2,3,4,5]:
            idx_short = (self.stim_labels[:,2]==i) * (self.stim_labels[:,3]==0)
            idx_long  = (self.stim_labels[:,2]==i) * (self.stim_labels[:,3]==1)
            neu_short.append(neu_cate[idx_short,:,:])
            neu_long.append(neu_cate[idx_long,:,:])
        idx_short = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        idx_long  = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_short.append(neu_cate[idx_short,:,:])
        neu_long.append(neu_cate[idx_long,:,:])
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, 0, -0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, 0, 0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')

    # roi mean response to normal stimulus.
    def roi_normal(self, ax, roi_id):
        self.plot_normal(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')

    # roi mean response to normal stimulus quantification.
    def roi_normal_box(self, ax, roi_id):
        self.plot_normal_box(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')
        
    # roi mean response to normal stimulus single trial heatmap.
    def roi_normal_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stimulus')
    
    # roi mean response to image change in short block.
    def roi_change_short(self, ax, roi_id):
        self.plot_change(ax, 0, roi_id=roi_id)
        ax.set_title('response to image change in short block')
    
    # roi mean response to image change in short block.
    def roi_change_long(self, ax, roi_id):
        self.plot_change(ax, 1, roi_id=roi_id)
        ax.set_title('response to image change in long block')
    
    # roi mean response to image change quantification.
    def roi_change_box(self, ax, roi_id):
        self.plot_change_box(ax, roi_id=roi_id)
        ax.set_title('response to image change')
    
    # roi mean response to image change single trial heatmap.
    def roi_change_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]<-1,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('response to image change')
    
    # roi mean response to short long context for all normal image.
    def roi_context_all(self, ax, roi_id):
        self.plot_context(ax, 5, roi_id=roi_id)
        ax.set_title('response to all normal image')
    
    # roi mean response to all stimulus average across trial for short.
    def roi_context_all_short_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (short)')

    # roi mean response to all stimulus average across trial for long.
    def roi_context_all_long_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (long)')
    
    # roi mean response to short long context for individual image.
    def roi_context_individual(self, axs, roi_id):
        titles = [
            'response to normal img#1',
            'response to normal img#2',
            'response to normal img#3',
            'response to normal img#4']
        for i in range(4):
            self.plot_context(axs[i], i+1, roi_id=roi_id)
            axs[i].set_title(titles[i])
            
    # roi mean response to short long context quantification
    def roi_context_box(self, ax, roi_id):
        self.plot_context_box(ax, roi_id=roi_id)
        ax.set_title('response to different image')
        
        
        
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        
    # excitory mean response to normal stimulus.
    def normal_exc(self, ax):
        self.plot_normal(ax, cate=-1)
        ax.set_title('excitory response to normal stimulus')
        
    # inhibitory mean response to normal stimulus.
    def normal_inh(self, ax):
        self.plot_normal(ax, cate=1)
        ax.set_title('inhibitory response to normal stimulus')
    
    # excitory mean response to all stimulus quantification.
    def normal_exc_box(self, ax):
        self.plot_normal_box(ax, cate=-1)
        ax.set_title('excitory response to normal stimulus')
        
    # inhibitory mean response to all stimulus quantification.
    def normal_inh_box(self, ax):
        self.plot_normal_box(ax, cate=1)
        ax.set_title('inhibitory response to normal stimulus')
    
    # excitory mean response to all stimulus single trial heatmap.
    def normal_exc_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('excitory single trial response to normal stimulus')
    
    # inhibitory mean response to all stimulus single trial heatmap.
    def normal_inh_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('inhibitory single trial response to normal stimulus')

    # excitory mean response to normal stimulus with epoch for short.
    def normal_epoch_exc_short(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch(ax, self.epoch_short, colors, cate=-1)
        ax.set_title('excitory response to normal stimulus with epoch (short)')
    
    # excitory mean response to normal stimulus with epoch for long.
    def normal_epoch_exc_long(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch(ax, self.epoch_long, colors, cate=-1)
        ax.set_title('excitory response to normal stimulus with epoch (long)')
        
    # inhibitory mean response to normal stimulus with epoch for short.
    def normal_epoch_inh_short(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch(ax, self.epoch_short, colors, cate=1)
        ax.set_title('inhibitory response to normal stimulus with epoch (short)')
    
    # inhibitory mean response to normal stimulus with epoch for long.
    def normal_epoch_inh_long(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch(ax, self.epoch_long, colors, cate=1)
        ax.set_title('inhibitory response to normal stimulus with epoch (long)')
    
    # excitory mean response to image change in short block.
    def change_exc_short(self, ax):
        self.plot_change(ax, 0, cate=-1)
        ax.set_title('excitory response to image change (short)')
    
    # excitory mean response to image change in short block.
    def change_exc_long(self, ax):
        self.plot_change(ax, 1, cate=-1)
        ax.set_title('excitory response to image change (long)')
        
    # inhibitory mean response to image change in short block.
    def change_inh_short(self, ax):
        self.plot_change(ax, 0, cate=1)
        ax.set_title('inhibitory response to image change (short)')
    
    # inhibitory mean response to image change in short block.
    def change_inh_long(self, ax):
        self.plot_change(ax, 1, cate=1)
        ax.set_title('inhibitory response to image change (long)')
        
    # excitory mean response to image change quantification.
    def change_exc_box(self, ax):
        self.plot_change_box(ax, cate=-1)
        ax.set_title('excitory response to image change')
        
    # inhibitory mean response to image change quantification.
    def change_inh_box(self, ax):
        self.plot_change_box(ax, cate=1)
        ax.set_title('inhibitory response to image change')

    # excitory mean response to image change single trial heatmap.
    def change_exc_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]<-1,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('excitory single trial response to image change')
    
    # inhibitory mean response to image change single trial heatmap.
    def change_inh_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]<-1,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('inhibitory single trial response to image change')
    
    # mean response to image change average across trial.
    def change_heatmap_neuron(self, ax):
        neu_seq = self.neu_seq[self.stim_labels[:,2]<-1,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to image change')
            
    # excitory mean response to short long context.
    def context_exc(self, axs):
        titles = [
            'excitory response to img#1',
            'excitory response to img#2',
            'excitory response to img#3',
            'excitory response to img#4',
            'excitory response to all image']
        for i in range(5):
            self.plot_context(axs[i], i+1, cate=-1)
            axs[i].set_title(titles[i])
    
    # inhibitory mean response to short long context.
    def context_inh(self, axs):
        titles = [
            'inhibitory response to img#1',
            'inhibitory response to img#2',
            'inhibitory response to img#3',
            'inhibitory response to img#4',
            'inhibitory response to all image']
        for i in range(5):
            self.plot_context(axs[i], i+1, cate=1)
            axs[i].set_title(titles[i])
        
    # excitory mean response to short long context quantification
    def context_exc_box(self, ax):
        self.plot_context_box(ax, cate=-1)
        ax.set_title('excitory response to different image')
    
    # inhibitory mean response to short long context quantification
    def context_inh_box(self, ax):
        self.plot_context_box(ax, cate=1)
        ax.set_title('inhibitory response to different image')
        
    # excitory mean response to short long context single trial heatmap.
    def context_exc_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('excitory response to different image')
    
    # inhibitory mean response to short long context single trial heatmap.
    def context_inh_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('inhibitory response to different image')
        
    # mean response to all stimulus average across trial for short.
    def context_all_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (short)')
    
    # mean response to all stimulus average across trial for long.
    def context_all_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (long)')
        
    # evoke context response comparison.
    def context_evoke(self, axs):
        win = [[-100,0], [0,250]]
        lbl = ['all', 'img#1', 'img#2', 'img#3', 'img#4']
        colors = [['#A4CB9E', '#F9C08A'], ['mediumseagreen', 'coral']]
        com_upper = 0
        com_lower = 0
        for excinh, cate in enumerate([-1,1]):
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            neu_short = []
            neu_long = []
            for w in range(2):
                idx_short = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
                idx_long  = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
                neu_short.append(neu_cate[idx_short,:,:])
                neu_long.append(neu_cate[idx_long,:,:])
                for img_idx in [2,3,4,5]:
                    idx_short = (self.stim_labels[:,2]==img_idx) * (self.stim_labels[:,3]==0)
                    idx_long  = (self.stim_labels[:,2]==img_idx) * (self.stim_labels[:,3]==1)
                    neu_short.append(neu_cate[idx_short,:,:])
                    neu_long.append(neu_cate[idx_long,:,:])
                for i in range(5):
                    [mean_short, sem_short] = get_mean_sem_win(
                        neu_short[i].reshape(-1, self.l_frames+self.r_frames),
                        self.neu_time, 0, win[w][0], win[w][1])
                    [mean_long, sem_long] = get_mean_sem_win(
                        neu_long[i].reshape(-1, self.l_frames+self.r_frames),
                        self.neu_time, 0, win[w][0], win[w][1])
                    upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
                    lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
                    com_upper = upper if upper > com_upper else com_upper
                    com_lower = lower if lower < com_lower else com_lower
                    axs[i].errorbar(
                        mean_short, mean_long,
                        xerr=sem_short, yerr=sem_long,
                        color=colors[w][excinh],
                        capsize=2, marker='o', linestyle='--',
                        markeredgecolor='white', markeredgewidth=1)
        for i in range(5):
            axs[i].plot([], color=colors[0][0], label='exc baseline')
            axs[i].plot([], color=colors[0][1], label='inh baseline')
            axs[i].plot([], color=colors[1][0], label='exc early evoked')
            axs[i].plot([], color=colors[1][1], label='inh early evoked')
            axs[i].plot(np.linspace(-1, 1), np.linspace(-1, 1), linestyle='--', color='grey')
            axs[i].set_xlim(com_lower, com_upper)
            axs[i].set_ylim(com_lower, com_upper)
            axs[i].tick_params(tick1On=False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].set_xlabel('df/f for short normal(z-scored)')
            axs[i].set_ylabel('df/f for long normal(z-scored)')
            axs[i].legend(loc='upper left')
            axs[i].set_title('evoked response comparison on '+lbl[i])
        
        
class plotter_VIPG8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        self.labels = np.ones_like(labels)
        
    # mean response to all stimulus.
    def normal(self, ax):
        self.plot_normal(ax, cate=1)
        ax.set_title('response to normal stimulus')
    
    # mean response to all stimulus quantification.
    def normal_box(self, ax):
        self.plot_normal_box(ax, cate=1)
        ax.set_title('response to normal stimulus')
    
    # mean response to image change in short block.
    def change_short(self, ax):
        self.plot_change(ax, 0, cate=1)
        ax.set_title('response to image change (short)')
    
    # mean response to image change in short block.
    def change_long(self, ax):
        self.plot_change(ax, 1, cate=1)
        ax.set_title('response to image change (long)')
    
    # mean response to image change quantification.
    def change_box(self, ax):
        self.plot_change_box(ax, cate=1)
        ax.set_title('response to image change')

    # mean response to image change average across trial.
    def change_heatmap_neuron(self, ax):
        neu_seq = self.neu_seq[self.stim_labels[:,2]<-1,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to image change')
        
    # mean response to short long context.
    def context(self, axs):
        titles = [
            'response to img#1',
            'response to img#2',
            'response to img#3',
            'response to img#4',
            'response to all image']
        for i in range(5):
            self.plot_context(axs[i], i+1, cate=1)
            axs[i].set_title(titles[i])
    
    # mean response to short long context quantification
    def context_box(self, ax):
        self.plot_context_box(ax, cate=1)
        ax.set_title('response to different image')
        
    # mean response to all stimulus average across trial for short.
    def context_all_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (short)')
    
    # mean response to all stimulus average across trial for long.
    def context_all_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (long)')

class plotter_L7G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        self.labels = -1*np.ones_like(labels)
        
    # mean response to all stimulus.
    def normal(self, ax):
        self.plot_normal(ax, cate=-1)
        ax.set_title('response to normal stimulus')
    
    # mean response to all stimulus quantification.
    def normal_box(self, ax):
        self.plot_normal_box(ax, cate=-1)
        ax.set_title('response to normal stimulus')
    
    # mean response to image change in short block.
    def change_short(self, ax):
        self.plot_change(ax, 0, cate=-1)
        ax.set_title('response to image change (short)')
    
    # mean response to image change in short block.
    def change_long(self, ax):
        self.plot_change(ax, 1, cate=-1)
        ax.set_title('response to image change (long)')
    
    # mean response to image change quantification.
    def change_box(self, ax):
        self.plot_change_box(ax, cate=-1)
        ax.set_title('response to image change')

    # mean response to image change average across trial.
    def change_heatmap_neuron(self, ax):
        neu_seq = self.neu_seq[self.stim_labels[:,2]<-1,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to image change')
        
    # mean response to short long context.
    def context(self, axs):
        titles = [
            'response to img#1',
            'response to img#2',
            'response to img#3',
            'response to img#4',
            'response to all image']
        for i in range(5):
            self.plot_context(axs[i], i+1, cate=-1)
            axs[i].set_title(titles[i])
    
    # mean response to short long context quantification
    def context_box(self, ax):
        self.plot_context_box(ax, cate=-1)
        ax.set_title('response to different image')
        
    # mean response to all stimulus average across trial for short.
    def context_all_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (short)')
    
    # mean response to all stimulus average across trial for long.
    def context_all_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to normal stimulus (long)')