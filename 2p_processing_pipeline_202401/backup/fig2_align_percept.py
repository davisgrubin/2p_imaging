#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode
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


# cut stim_seq based on the shortest one.
def cut_stim_seq(stim_seq):
    min_len = np.min([s.shape[1] for s in stim_seq])
    stim_seq = [s[:,:min_len] for s in stim_seq]
    return stim_seq


# extract response around pre perturbation stimulus with outcome.
def get_pre_pert_stim_response(
        neural_trials, outcome_state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_outcome = neural_trials[trials][outcome_state]
        time_stim_seq = neural_trials[str(trials)]['time_stim_seq']
        if (not np.isnan(time_outcome[0]) and
            not np.isnan(time_stim_seq[0,0]) and
            time_stim_seq.shape[1] > 4
            ):
            idx = np.argmin(np.abs(time - time_stim_seq[0,2]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # get stim timestamps
                stim_seq.append(np.array(
                    [0, time_stim_seq[1,2]-time_stim_seq[0,2]]))
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = np.mean(stim_seq, axis=0)
    return [neu_seq, neu_time, stim_seq]


# extract response around pre perturbation stimulus with outcome.
def get_post_pert_stim_response_side(
        neural_trials_side, outcome_state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    # loop over trials.
    for trials in neural_trials_side.keys():
        # read trial data.
        fluo = neural_trials_side[trials]['dff']
        time = neural_trials_side[trials]['time']
        time_outcome = neural_trials_side[trials][outcome_state]
        time_stim_seq = neural_trials_side[str(trials)]['time_stim_seq']
        if (not np.isnan(time_outcome[0]) and
            not np.isnan(time_stim_seq[0,0]) and
            time_stim_seq.shape[1] > 4
            ):
            idx = np.argmin(np.abs(time - time_stim_seq[0,2]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # get stim timestamps
                stim_seq.append(time_stim_seq[:,2:] - time_stim_seq[0,2])
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = cut_stim_seq(stim_seq)
    stim_seq = np.concatenate(
        [np.expand_dims(s, axis=0) for s in stim_seq],
        axis=0)
    stim_seq = np.mean(stim_seq, axis=0)
    return [neu_seq, neu_time, stim_seq]


# extract response for post perturbation.
def separate_left_right_trials(neural_trials):
    neural_trials_left = dict()
    neural_trials_right = dict()
    for i in range(len(neural_trials)):
        if neural_trials[str(i)]['trial_types']==1:
            neural_trials_left[str(i)] = neural_trials[str(i)]
        if neural_trials[str(i)]['trial_types']==2:
            neural_trials_right[str(i)] = neural_trials[str(i)]
    return [neural_trials_left, neural_trials_right]


# get ROI color from label.
def get_roi_label_color(labels, roi_id):
    if labels[roi_id] == -1:
        cate = 'excitory'
        color1 = 'mediumseagreen'
        color2 = 'turquoise'
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
    if labels[roi_id] == 0:
        cate = 'unsure'
        color1 = 'violet'
        color2 = 'dodgerblue'
        cmap = LinearSegmentedColormap.from_list(
            'unsure', ['white', 'dodgerblue', 'black'])
    if labels[roi_id] == 1:
        cate = 'inhibitory'
        color1 = 'brown'
        color2 = 'coral'
        cmap = LinearSegmentedColormap.from_list(
            'inhibitory', ['white', 'coral', 'black'])
    return cate, color1, color2, cmap


# adjust layout for grand average.
def adjust_layout_grand(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')
    

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time since first grating after omission (ms)')


class plotter_VIPTD_G8_align_perc:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials, labels
            ):
        self.pre_l_frames = 30
        self.pre_r_frames = 50
        self.post_l_frames = 30
        self.post_r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        self.labels = labels
        [self.neural_trials_left,
         self.neural_trials_right] = separate_left_right_trials(
             neural_trials)
        [self.neu_seq_pre_reward,
         self.neu_time_pre_reward,
         self.pre_reward_seq] = get_pre_pert_stim_response(
                 neural_trials, 'time_reward',
                 self.pre_l_frames, self.pre_r_frames)
        [self.neu_seq_pre_punish,
         self.neu_time_pre_punish,
         self.pre_punish_seq] = get_pre_pert_stim_response(
                 neural_trials, 'time_punish',
                 self.pre_l_frames, self.pre_r_frames)
        [self.neu_seq_left_reward,
         self.neu_time_left_reward,
         self.stim_seq_left_reward] = get_post_pert_stim_response_side(
                 self.neural_trials_left, 'time_reward',
                 self.post_l_frames, self.post_r_frames)
        [self.neu_seq_right_reward,
         self.neu_time_right_reward,
         self.stim_seq_right_reward] = get_post_pert_stim_response_side(
                 self.neural_trials_right, 'time_reward',
                 self.post_l_frames, self.post_r_frames)
        [self.neu_seq_left_punish,
         self.neu_time_left_punish,
         self.stim_seq_left_punish] = get_post_pert_stim_response_side(
                 self.neural_trials_left, 'time_punish',
                 self.post_l_frames, self.post_r_frames)
        [self.neu_seq_right_punish,
         self.neu_time_right_punish,
         self.stim_seq_right_punish] = get_post_pert_stim_response_side(
                 self.neural_trials_right, 'time_punish',
                 self.post_l_frames, self.post_r_frames)

    # mean response for pre perturbation.
    def plot_pre_align(self, ax, state, neu_seq, neu_time, outcome_seq):
        exc_all = neu_seq[:,self.labels==-1,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames)
        inh_all = neu_seq[:,self.labels==1,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames)
        mean_exc = np.mean(exc_all, axis=0)
        mean_inh = np.mean(inh_all, axis=0)
        sem_exc = sem(exc_all, axis=0)
        sem_inh = sem(inh_all, axis=0)
        upper = np.max([mean_exc, mean_inh]) + np.max([sem_exc, sem_inh])
        lower = np.min([mean_exc, mean_inh]) - np.max([sem_exc, sem_inh])
        ax.fill_between(
            outcome_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label=state)
        ax.plot(
            neu_time,
            mean_exc,
            color='turquoise', label='exc')
        ax.fill_between(
            neu_time,
            mean_exc - sem_exc,
            mean_exc + sem_exc,
            color='turquoise', alpha=0.2)
        ax.plot(
            neu_time,
            mean_inh,
            color='coral', label='inh')
        ax.fill_between(
            neu_time,
            mean_inh - sem_inh,
            mean_inh + sem_inh,
            color='coral', alpha=0.2)
        adjust_layout_grand(ax)
        ax.legend(loc='upper right')
        ax.set_xlim([np.min(neu_time), np.max(neu_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    # pre perturbation response with reward.
    def pre_reward(self, ax):
        self.plot_pre_align(
            ax, 'reward',
            self.neu_seq_pre_reward, self.neu_time_pre_reward, self.pre_reward_seq)
        ax.set_xlabel('time since 2nd pre pert stim (ms)')
        ax.set_title('response to pre pert stim with reward')
    
    # pre perturbation response with punish.
    def pre_punish(self, ax):
        self.plot_pre_align(
            ax, 'punish',
            self.neu_seq_pre_punish, self.neu_time_pre_punish, self.pre_punish_seq)
        ax.set_xlabel('time since 2nd pre pert stim (ms)')
        ax.set_title('response to pre pert stim with punish')
    
    # mean response heatmap on given outcome.
    def plot_mean_outcome_heatmap(self, ax, neu_seq, neu_time):
        mean = np.mean(neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(neu_time==0)[0][0]
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'fix', ['white','violet', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id (sorted)')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.set_xticks([0, zero, len(neu_time)])
        ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])

    # mean response heatmap on reward.
    def reward_heatmap(self, ax):
        self.plot_mean_outcome_heatmap(
            ax, self.neu_seq_pre_reward, self.neu_time_pre_reward)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response heatmap to reward')
    
    # mean response heatmap on reward.
    def punish_heatmap(self, ax):
        self.plot_mean_outcome_heatmap(
            ax, self.neu_seq_pre_punish, self.neu_time_pre_punish)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response heatmap to punish')
    
    # mean response for post perturbation.
    def plot_post_align(self, ax, neu_seq, neu_time, stim_seq):
        exc_all = neu_seq[:,self.labels==-1,:].reshape(
            -1, self.post_l_frames+self.post_r_frames)
        inh_all = neu_seq[:,self.labels==1,:].reshape(
            -1, self.post_l_frames+self.post_r_frames)
        mean_exc = np.mean(exc_all, axis=0)
        mean_inh = np.mean(inh_all, axis=0)
        sem_exc = sem(exc_all, axis=0)
        sem_inh = sem(inh_all, axis=0)
        upper = np.max([mean_exc, mean_inh]) + np.max([sem_exc, sem_inh])
        lower = np.min([mean_exc, mean_inh]) - np.max([sem_exc, sem_inh])
        for i in range(stim_seq.shape[1]):
            ax.fill_between(
                stim_seq[:,i].reshape(-1),
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='silver', alpha=0.25, step='mid', label='stim')
        ax.plot(
            neu_time,
            mean_exc,
            color='turquoise', label='exc')
        ax.fill_between(
            neu_time,
            mean_exc - sem_exc,
            mean_exc + sem_exc,
            color='turquoise', alpha=0.2)
        ax.plot(
            neu_time,
            mean_inh,
            color='coral', label='inh')
        ax.fill_between(
            neu_time,
            mean_inh - sem_inh,
            mean_inh + sem_inh,
            color='coral', alpha=0.2)
        adjust_layout_grand(ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:])
        ax.set_xlim([np.min(neu_time), np.max(neu_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        
    # post perturbation response for left with reward.
    def post_left_reward(self, ax):
        self.plot_post_align(
            ax,
            self.neu_seq_left_reward, self.neu_time_left_reward,
            self.stim_seq_left_reward)
        ax.set_xlabel('time since perturbation (ms)')
        ax.set_title('response to post pert stim with reward (left side)')
    
    # post perturbation response for left with punish.
    def post_left_punish(self, ax):
        self.plot_post_align(
            ax,
            self.neu_seq_left_punish, self.neu_time_left_punish,
            self.stim_seq_left_punish)
        ax.set_xlabel('time since perturbation (ms)')
        ax.set_title('response to post pert stim with punish (left side)')
    
    # post perturbation response for right with reward.
    def post_right_reward(self, ax):
        self.plot_post_align(
            ax,
            self.neu_seq_right_reward, self.neu_time_right_reward,
            self.stim_seq_right_reward)
        ax.set_xlabel('time since perturbation (ms)')
        ax.set_title('response to post pert stim with reward (right side)')
    
    # post perturbation response for right with punish.
    def post_right_punish(self, ax):
        self.plot_post_align(
            ax,
            self.neu_seq_right_punish, self.neu_time_right_punish,
            self.stim_seq_right_punish)
        ax.set_xlabel('time since perturbation (ms)')
        ax.set_title('response to post pert stim with punish (right side)')


    # ROI mean response for all stimulus.
    
    # ROI mean response for pre perturbation.
    def roi_pre_pert(self, ax, roi_id):
        mean_reward = np.mean(self.neu_seq_pre_reward[:,roi_id,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames), axis=0)
        mean_punish = np.mean(self.neu_seq_pre_punish[:,roi_id,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames), axis=0)
        sem_reward = sem(self.neu_seq_pre_reward[:,roi_id,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames), axis=0)
        sem_punish = sem(self.neu_seq_pre_punish[:,roi_id,:].reshape(
            -1, self.pre_l_frames+self.pre_r_frames), axis=0)
        upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
        lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
        _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        ax.fill_between(
            self.pre_reward_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='silver', alpha=0.25, step='mid', label='2nd pre stim')
        ax.plot(
            self.neu_time_pre_reward,
            mean_reward,
            color=color1, label='reward')
        ax.fill_between(
            self.neu_time_pre_reward,
            mean_reward - sem_reward,
            mean_reward + sem_reward,
            color=color1, alpha=0.2)
        ax.plot(
            self.neu_time_pre_punish,
            mean_punish,
            color=color2, label='punish')
        ax.fill_between(
            self.neu_time_pre_punish,
            mean_punish - sem_punish,
            mean_punish + sem_punish,
            color=color2, alpha=0.2)
        adjust_layout_grand(ax)
        ax.legend(loc='upper right')
        ax.set_xlim([np.min(self.neu_time_pre_reward), np.max(self.neu_time_pre_reward)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title('response to 2nd pre pert stim')

    # ROI mean response for post perturbation.
    def roi_plot_post_pert_side(
            self, ax, roi_id,
            neu_seq_reward, neu_time_reward, stim_seq_reward,
            neu_seq_punish, neu_time_punish, stim_seq_punish,
            ):
        mean_reward = np.mean(neu_seq_reward[:,roi_id,:].reshape(
            -1, self.post_l_frames+self.post_r_frames), axis=0)
        mean_punish = np.mean(neu_seq_punish[:,roi_id,:].reshape(
            -1, self.post_l_frames+self.post_r_frames), axis=0)
        sem_reward = sem(neu_seq_reward[:,roi_id,:].reshape(
            -1, self.post_l_frames+self.post_r_frames), axis=0)
        sem_punish = sem(neu_seq_punish[:,roi_id,:].reshape(
            -1, self.post_l_frames+self.post_r_frames), axis=0)
        upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
        lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
        _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        for i in range(stim_seq_reward.shape[1]):
            ax.fill_between(
                stim_seq_reward[:,i].reshape(-1),
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='silver', alpha=0.25, step='mid', label='stim')
        ax.plot(
            neu_time_reward,
            mean_reward,
            color=color1, label='reward')
        ax.fill_between(
            neu_time_reward,
            mean_reward - sem_reward,
            mean_reward + sem_reward,
            color=color1, alpha=0.2)
        ax.plot(
            neu_time_reward,
            mean_punish,
            color=color2, label='punish')
        ax.fill_between(
            neu_time_reward,
            mean_punish - sem_punish,
            mean_punish + sem_punish,
            color=color2, alpha=0.2)
        adjust_layout_grand(ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-3:], labels[-3:])
        ax.set_xlim([np.min(neu_time_reward), np.max(neu_time_reward)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])

    # ROI response to post perturbation stimulus for left.
    def roi_post_pert_left(self, ax, roi_id):
        self.plot_post_pert_side(
                ax, roi_id,
                self.neu_seq_left_reward,
                self.neu_time_left_reward,
                self.stim_seq_left_reward,
                self.neu_seq_left_punish,
                self.neu_time_left_punish,
                self.stim_seq_left_punish)
        ax.set_title('response to post pert stim (left side)')
        
    # ROI response to post perturbation stimulus for right.
    def roi_post_pert_right(self, ax, roi_id):
        self.plot_post_pert_side(
                ax, roi_id,
                self.neu_seq_right_reward,
                self.neu_time_right_reward,
                self.stim_seq_right_reward,
                self.neu_seq_right_punish,
                self.neu_time_right_punish,
                self.stim_seq_right_punish)
        ax.set_title('response to post pert stim (right side)')
        