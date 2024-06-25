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
    neu_time = [nt.reshape(1,-1) for nt in neu_time]
    neu_seq  = np.concatenate(neu_seq, axis=0)
    neu_time = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    return neu_seq, neu_time


# extract response around all licking.
def get_lick_response(
        neural_trials, lick_state,
        l_frames, r_frames):
    # initialize list.
    neu_seq_lick  = []
    neu_time_lick = []
    lick_direc = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_licking = neural_trials[trials][lick_state]
        for lick_idx in range(time_licking.shape[1]):
            if not np.isnan(time_licking[0,lick_idx]):
                idx = np.argmin(np.abs(time - time_licking[0,lick_idx]))
                if idx > l_frames and idx < len(time)-r_frames:
                    # signal response.
                    f = fluo[:, idx-l_frames : idx+r_frames]
                    f = np.expand_dims(f, axis=0)
                    neu_seq_lick.append(f)
                    # signal time stamps.
                    t = time[idx-l_frames : idx+r_frames] - time[idx]
                    neu_time_lick.append(t)
                    # licking direction 0-left 1-right.
                    lick_direc.append(time_licking[1,lick_idx])
    neu_seq_lick, neu_time_lick = align_neu_seq_utils(neu_seq_lick, neu_time_lick)
    lick_direc = np.array(lick_direc)
    return [neu_seq_lick, neu_time_lick, lick_direc]


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
    ax.legend(loc='upper right')


# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
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
        self.l_frames = 40
        self.r_frames = 50
        self.neural_trials = neural_trials
        self.labels = labels
        [self.neu_seq_lick_all,
         self.neu_time_lick_all,
         self.lick_direc_all] = get_lick_response(
                 neural_trials, 'trial_lick',
                 self.l_frames, self.r_frames)
        [self.neu_seq_lick_reaction,
         self.neu_time_lick_reaction,
         self.lick_direc_reaction] = get_lick_response(
                 neural_trials, 'trial_reaction',
                 self.l_frames, self.r_frames)
        [self.neu_seq_lick_decision,
         self.neu_time_lick_decision,
         self.lick_direc_decision] = get_lick_response(
                 neural_trials, 'trial_decision',
                 self.l_frames, self.r_frames)

    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])

    def plot_lick(
            self, ax,
            neu_seq_lick, neu_time_lick, lick_direc,
            cate=None, roi_id=None
            ):
        if cate != None:
            neu_cate = neu_seq_lick[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq_lick[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)

        neu_seq_left = neu_cate[lick_direc==0, :, :].copy()
        neu_seq_left = neu_seq_left.reshape(-1, self.l_frames+self.r_frames)
        neu_seq_right = neu_cate[lick_direc==1, :, :].copy()
        neu_seq_right = neu_seq_right.reshape(-1, self.l_frames+self.r_frames)
        mean_left  = np.mean(neu_seq_left, axis=0)
        mean_right = np.mean(neu_seq_right, axis=0)
        sem_left   = sem(neu_seq_left, axis=0)
        sem_right  = sem(neu_seq_right, axis=0)
        upper = np.max([np.max(mean_left), np.max(mean_right)]) + \
                np.max([np.max(sem_left), np.max(sem_right)])
        lower = np.min([np.min(mean_left), np.min(mean_right)]) - \
                np.max([np.max(sem_left), np.max(sem_right)])
        ax.axvline(0, color='gold', lw=1, label='licking', linestyle='--')
        self.plot_mean_sem(ax, neu_time_lick, mean_left, sem_left, 'mediumseagreen', 'left')
        self.plot_mean_sem(ax, neu_time_lick, mean_right, sem_right, 'coral', 'right')
        adjust_layout_grand(ax)
        ax.set_xlim([np.min(neu_time_lick), np.max(neu_time_lick)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since licking event (ms)')

    def plot_heatmap_neuron(self, ax, neu_seq, neu_time):
        if len(neu_seq) > 0:
            _, _, _, cmap_exc = get_roi_label_color([-1], 0)
            _, _, _, cmap_inh = get_roi_label_color([1], 0)
            mean = np.mean(neu_seq, axis=0)
            zero = np.argmin(np.abs(neu_time - 0))
            smoothed_mean = np.array([np.convolve(row, np.ones(5)/5, mode='same') for row in mean])
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
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

class plotter_VIPTD_G8_align_beh(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)

    # excitory mean response to all licking events.
    def all_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all,
            cate=-1)
        ax.set_title('excitory response to all licking events')

    # inhibitory mean response to all licking events.
    def all_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all,
            cate=1)
        ax.set_title('inhibitory response to all licking events')

    # response heatmap to all licking events average across trials.
    def all_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_all, self.neu_time_lick_all)
        ax.set_title('response heatmap to all licking events')

    # excitory mean response to reaction licking.
    def reaction_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_reaction,
            self.neu_time_lick_reaction,
            self.lick_direc_reaction,
            cate=-1)
        ax.set_title('excitory response to reaction licking')

    # inhibitory mean response to reaction licking.
    def reaction_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_reaction,
            self.neu_time_lick_reaction,
            self.lick_direc_reaction,
            cate=1)
        ax.set_title('inhibitory response to reaction licking')

    # response heatmap to reaction licking average across trials.
    def reaction_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_reaction, self.neu_time_lick_reaction)
        ax.set_title('response heatmap to reaction licking')

    # excitory mean response to decision licking.
    def decision_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision,
            cate=-1)
        ax.set_title('excitory response to decision licking')

    # inhibitory mean response to decision licking.
    def decision_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision,
            cate=1)
        ax.set_title('inhibitory response to decision licking')

    # response heatmap to decision licking average across trials.
    def decision_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_decision, self.neu_time_lick_decision)
        ax.set_title('response heatmap to decision licking')


    def plot_roi_lick(self, ax, roi_id, neu_seq_lick, neu_time_lick, lick_direc):
        mean_left = np.mean(neu_seq_lick[lick_direc==0, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        mean_right = np.mean(neu_seq_lick[lick_direc==1, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        sem_left = sem(neu_seq_lick[lick_direc==0, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        sem_right = sem(neu_seq_lick[lick_direc==1, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([np.max(mean_left), np.max(mean_right)]) + \
                np.max([np.max(sem_left), np.max(sem_right)])
        lower = np.min([np.min(mean_left), np.min(mean_right)]) - \
                np.max([np.max(sem_left), np.max(sem_right)])
        ax.axvline(0, color='black', lw=1, label='licking', linestyle='--')
        ax.plot(
            neu_time_lick,
            mean_left,
            color='mediumseagreen', label='left licking')
        ax.fill_between(
            neu_time_lick,
            mean_left - sem_left,
            mean_left + sem_left,
            color='mediumseagreen', alpha=0.2)
        ax.plot(
            neu_time_lick,
            mean_right,
            color='coral', label='right licking')
        ax.fill_between(
            neu_time_lick,
            mean_right - sem_right,
            mean_right + sem_right,
            color='coral', alpha=0.2)
        adjust_layout_grand(ax)
        ax.set_xlim([np.min(neu_time_lick), np.max(neu_time_lick)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since licking event (ms)')

    def plot_roi_lick_box(self, ax, roi_id, neu_seq_lick, neu_time_lick, lick_direc):
        win_base  = [-100,0]
        win_early = [0,250]
        win_late  = [250,500]
        win = [win_base, win_early, win_late]
        pos = [-0.1, 0, 0.1]
        center = [1,2]
        colors = ['#A4CB9E', '#9DB4CE', '#EDA1A4']
        roi_left = neu_seq_lick[lick_direc==0, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        roi_right = neu_seq_lick[lick_direc==1, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        roi_neu = [roi_left, roi_right]
        for i in range(2):
            for j in range(3):
                [roi_mean, roi_sem] = get_mean_sem_win(
                    roi_neu[i], neu_time_lick,
                    0, win[j][0], win[j][1])
                ax.errorbar(
                    center[i] + pos[j],
                    roi_mean,
                    roi_sem,
                    color=colors[j],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        stages = ['baseline: -100-0 ms', 'early: 0-250 ms', 'late: 250-500 ms']
        for i in range(3):
            ax.plot([], c=colors[i], label=stages[i])
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.set_xticks(center)
        ax.set_xticklabels(['left', 'right'])
        ax.set_xlim([0.5,4])
        ax.legend(loc='upper right')

    # ROI response to all licking.
    def roi_lick_all(self, ax, roi_id):
        self.plot_roi_lick(
            ax, roi_id,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all)
        ax.set_title('response for all licking events on two sides')

    # ROI response to all licking quantification.
    def roi_lick_all_box(self, ax, roi_id):
        self.plot_roi_lick_box(
            ax, roi_id,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all)
        ax.set_title('response for all licking events on two sides')

    # ROI response to decision licking.
    def roi_lick_decision(self, ax, roi_id):
        self.plot_roi_lick(
            ax, roi_id,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision)
        ax.set_title('response for decision on two sides')

    # ROI response to decision licking quantification.
    def roi_lick_decision_box(self, ax, roi_id):
        self.plot_roi_lick_box(
            ax, roi_id,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision)
        ax.set_title('response for decision on two sides')
