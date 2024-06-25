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


# get stimulus idx for pre/post perturbations.
def get_stim_idx(time_stim_seq, mode):
    if np.isnan(time_stim_seq[0,0]) or time_stim_seq.shape[1]<3:
        stim_idx = []
        stim_isi = []
    else:
        if mode == 'onset':
            stim_idx = np.array([0])
            stim_isi = []
        if mode == 'all':
            stim_idx = np.arange(0, time_stim_seq.shape[1])
            stim_isi = []
        if mode == 'pre':
            stim_idx = np.arange(0, 3)
            stim_isi = np.mean(time_stim_seq[0,1:3] - time_stim_seq[1,0:2])
        if mode == 'post':
            if time_stim_seq.shape[1]<4:
                stim_idx = []
                stim_isi = []
            else:
                stim_idx = np.arange(3, time_stim_seq.shape[1])
                stim_isi = np.mean(time_stim_seq[0,3:] - time_stim_seq[1,2:-1])
        if mode == 'perturb':
            stim_idx = np.array([2])
            stim_isi = []
    return stim_idx, stim_isi


# extract response around pre perturbation stimulus with outcome.
def get_stim_response(
        neural_trials, mode,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    trial_types = []
    trial_isi = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_stim_seq = neural_trials[trials]['trial_stim_seq']
        trial_punish = neural_trials[str(trials)]['trial_punish']
        trial_reward = neural_trials[str(trials)]['trial_reward']
        stim_idx, stim_isi = get_stim_idx(time_stim_seq, mode)
        if len(stim_idx) > 0:
            for stim_i in range(len(stim_idx)):
                idx = np.argmin(np.abs(time - time_stim_seq[0,stim_i]))
                if idx > l_frames and idx < len(time)-r_frames:
                    # signal response.
                    f = fluo[:, idx-l_frames : idx+r_frames]
                    f = np.expand_dims(f, axis=0)
                    neu_seq.append(f)
                    # signal time stamps.
                    t = time[idx-l_frames : idx+r_frames] - time[idx]
                    neu_time.append(t)
                    # stim timestamps.
                    stim_seq.append(np.array(
                        [0, time_stim_seq[1,stim_i]-time_stim_seq[0,stim_i]]))
                    # trial type.
                    trial_types.append(neural_trials[trials]['trial_types'])
                    # trial isi.
                    trial_isi.append(stim_isi)
                    # outcome.
                    if not np.isnan(trial_punish[0]):
                        trial_outcome = -1
                    elif not np.isnan(trial_reward[0]):
                        trial_outcome = 1
                    else:
                        trial_outcome = 0
                    outcome.append(trial_outcome)
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = np.mean(stim_seq, axis=0)
    trial_types = np.array(trial_types).reshape(-1)
    trial_isi = np.array(trial_isi).reshape(-1)
    return [neu_seq, neu_time, stim_seq, trial_types, trial_isi, outcome]


# extract response around outcome.
def get_outcome_response(
        neural_trials, state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    outcome_seq = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_outcome = neural_trials[str(trials)][state]
        # compute stimulus start point in ms.
        if not np.isnan(time_outcome[0]):
            idx = np.argmin(np.abs(time - time_outcome[0]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # get outcome timestamps
                outcome_seq.append(time_outcome - time_outcome[0])
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome_seq = np.mean(outcome_seq, axis=0)
    return [neu_seq, neu_time, outcome_seq]


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
        self.l_frames = 30
        self.r_frames = 50
        self.neural_trials = neural_trials
        self.labels = labels
        [self.neu_seq_onset, self.neu_time_onset,
         self.stim_seq_onset, _, _, self.outcome_onset] = get_stim_response(
                neural_trials, 'onset', self.l_frames, self.r_frames)
        [self.neu_seq_all, self.neu_time_all,
         self.stim_seq_all,
         _, self.trial_isi, self.outcome_all] = get_stim_response(
                neural_trials, 'all', self.l_frames, self.r_frames)
        [self.neu_seq_pre, self.neu_time_pre,
         self.stim_seq_pre,
         _, self.trial_isi_pre, self.outcome_pre] = get_stim_response(
                neural_trials, 'pre', self.l_frames, self.r_frames)
        [self.neu_seq_perturb, self.neu_time_perturb,
         self.stim_seq_perturb,
         self.trial_types_perturb, self.trial_isi_perturb,
         self.outcome_perturb] = get_stim_response(
                neural_trials, 'perturb', self.l_frames, self.r_frames)
        [self.neu_seq_post, self.neu_time_post,
         self.stim_seq_post,
         self.trial_types_post, self.trial_isi_post,
         self.outcome_post] = get_stim_response(
                neural_trials, 'post', self.l_frames, self.r_frames)
        [self.neu_seq_reward,
         self.neu_time_reward,
         self.reward_seq] = get_outcome_response(
             neural_trials, 'trial_reward',
             self.l_frames, self.r_frames)
        [self.neu_seq_punish,
         self.neu_time_punish,
         self.punish_seq] = get_outcome_response(
             neural_trials, 'trial_punish',
             self.l_frames, self.r_frames)

    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])

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

    def plot_all(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_all[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_all[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        mean_neu = np.mean(neu_cate.reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        sem_neu = sem(neu_cate.reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq_all,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time_all, mean_neu, sem_neu, color, 'dff')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_onset(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_onset[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_onset[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        mean_neu = np.mean(neu_cate.reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        sem_neu = sem(neu_cate.reshape(
            -1, self.l_frames+self.r_frames), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq_pre,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        ax.fill_between(
            self.stim_seq_pre + 1*(np.mean(self.trial_isi_pre)+self.stim_seq_pre[1]),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        ax.fill_between(
            self.stim_seq_pre + 2*(np.mean(self.trial_isi_pre)+self.stim_seq_pre[1]),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        self.plot_mean_sem(ax, self.neu_time_onset, mean_neu, sem_neu, color, 'dff')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim onset (ms)')

    def plot_pre(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_pre[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_pre[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        mean_neu = np.mean(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_neu = sem(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq_pre,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        ax.fill_between(
            self.stim_seq_pre - np.mean(self.trial_isi_pre) - self.stim_seq_pre[1],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        ax.fill_between(
            self.stim_seq_pre + np.mean(self.trial_isi_pre) + self.stim_seq_pre[1],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        self.plot_mean_sem(ax, self.neu_time_pre, mean_neu, sem_neu, color, 'dff')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since 2nd pre stim (ms)')


    def plot_pert(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_perturb[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_perturb[:,roi_id,:], axis=1)
        mean_all = np.mean(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_all = sem(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        neu_short  = neu_cate[self.trial_types_perturb==1,:,:].reshape(-1, self.l_frames+self.r_frames)
        mean_short = np.mean(neu_short, axis=0)
        sem_short  = sem(neu_short, axis=0)
        neu_long  = neu_cate[self.trial_types_perturb==2,:,:].reshape(-1, self.l_frames+self.r_frames)
        mean_long = np.mean(neu_long, axis=0)
        sem_long = sem(neu_long, axis=0)
        upper = np.max([mean_all, mean_short, mean_short]) + \
                np.max([sem_all, sem_short, sem_short])
        lower = np.min([mean_all, mean_short, mean_short]) - \
                np.max([sem_all, sem_short, sem_short])
        ax.fill_between(
            self.stim_seq_perturb,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='silver', alpha=0.25, step='mid', label='1st pert stim')
        self.plot_mean_sem(ax, self.neu_time_perturb, mean_all, sem_all,
                      'grey', 'all')
        self.plot_mean_sem(ax, self.neu_time_perturb, mean_short, sem_short,
                      'mediumseagreen', 'short ISI')
        self.plot_mean_sem(ax, self.neu_time_perturb, mean_long, sem_long,
                      'coral', 'long ISI')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since 1st pert stim (ms)')

    def plot_post_isi(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_post[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_post[:,roi_id,:], axis=1)
        bins = [0, 205, 300, 400, 500, 600, 700, 800, 1000]
        isi_labels = ['100-200', '200-300', '300-400', '400-500',
                      '500-600', '600-700', '700=800', '800-900']
        bin_idx = np.digitize(self.trial_isi_post, bins)-1
        colors = plt.cm.gist_rainbow(np.arange(len(bins))/len(bins))
        m = []
        s = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                neu_post = neu_cate[bin_idx==i,:,:].reshape(-1, self.l_frames+self.r_frames)
                neu_mean = np.mean(neu_post, axis=0)
                neu_sem = sem(neu_post, axis=0)
                self.plot_mean_sem(
                    ax, self.neu_time_post, neu_mean, neu_sem,
                    colors[i], isi_labels[i])
                m.append(neu_mean)
                s.append(neu_sem)
        upper = np.max(m) + np.max(s)
        lower = np.min(m) - np.max(s)
        ax.fill_between(
            self.stim_seq_post,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        adjust_layout_grand(ax)
        ax.set_xlim([np.min(self.neu_time_post), np.max(self.neu_time_post)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])

    def plot_outcome(
            self, ax,
            neu_seq, neu_time, outcome,
            cate=None, roi_id=None):
        if cate != None:
            neu_cate = neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        mean_neu = np.mean(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_neu = sem(neu_cate.reshape(-1, self.l_frames+self.r_frames), axis=0)
        self.plot_mean_sem(ax, neu_time, mean_neu, sem_neu, color, 'dff')
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            outcome,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='outcome')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
class plotter_VIPTD_G8_align_perc(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)

    # excitory response to all stimulus.
    def all_exc(self, ax):
        self.plot_all(ax, cate=-1)
        ax.set_title('excitory response to all stim')

    # inhibitory response to all stimulus.
    def all_inh(self, ax):
        self.plot_all(ax, cate=1)
        ax.set_title('inhibitory response to all stim')

    # response to all stimulus heatmap average across trials.
    def all_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_all, self.neu_time_all)
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to all stim')

    # excitory response to stimulus onset.
    def onset_exc(self, ax):
        self.plot_onset(ax, cate=-1)
        ax.set_title('excitory response to stim onset')

    # inhibitory response to stimulus onset.
    def onset_inh(self, ax):
        self.plot_onset(ax, cate=1)
        ax.set_title('inhibitory response to stim onset')

    # response to stimulus onset heatmap average across trials.
    def onset_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_onset, self.neu_time_onset)
        ax.set_xlabel('time since stim onset (ms)')
        ax.set_title('response to stim onset')

    # excitory response to pre perturbation stimulus.
    def pre_exc(self, ax):
        self.plot_pre(ax, cate=-1)
        ax.set_title('excitory response to pre pert stim')

    # inhibitory response to perturbation stimulus.
    def pre_inh(self, ax):
        self.plot_pre(ax, cate=1)
        ax.set_xlabel('time since 2nd pre stim (ms)')
        ax.set_title('inhibitory response to pre pert stim')

    # response to perturbation stimulus heatmap average across trials.
    def pre_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_pre, self.neu_time_pre)
        ax.set_title('response to pre pert stim')

    # excitory response to perturbation stimulus.
    def pert_exc(self, ax):
        self.plot_pert(ax, cate=-1)
        ax.set_title('excitory response to 1st pert stim')

    # inhibitory response to perturbation stimulus.
    def pert_inh(self, ax):
        self.plot_pert(ax, cate=1)
        ax.set_title('inhibitory response to 1st pert stim')

    # response to perturbation stimulus heatmap average across trials.
    def pert_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_all, self.neu_time_all)
        ax.set_xlabel('time since 1st pert stim (ms)')
        ax.set_title('response to 1st pert stim')

    # excitory response to post perturbation stimulus with isi.
    def post_isi_exc(self, ax):
        self.plot_post_isi(ax, cate=-1)
        ax.set_title('excitory response to post pert stim')

    # inhibitory response to post perturbation stimulus with isi.
    def post_isi_inh(self, ax):
        self.plot_post_isi(ax, cate=1)
        ax.set_title('inhibitory response to post pert stim')

    # excitory mean response to reward.
    def reward_exc(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward,
            self.reward_seq, cate=-1)
        ax.set_xlabel('time since reward start (ms)')
        ax.set_title('excitory response to reward')

    # inhibitory mean response to reward.
    def reward_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward,
            self.reward_seq, cate=1)
        ax.set_xlabel('time since reward start (ms)')
        ax.set_title('inhibitory response to reward')

    # response to reward heatmap average across trials.
    def reward_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_reward, self.neu_time_reward)
        ax.set_xlabel('time since reward start (ms)')
        ax.set_title('response to reward')

    # excitory mean response to punish.
    def punish_exc(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish,
            self.punish_seq, cate=-1)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('excitory response to punish')

    # inhibitory mean response to punish.
    def punish_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish,
            self.punish_seq, cate=1)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('inhibitory response to punish')

    # response to punish heatmap average across trials.
    def punish_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_punish, self.neu_time_punish)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('response to punish')
