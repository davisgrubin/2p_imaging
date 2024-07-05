#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

from modules.Alignment import get_outcome_response
from modules.Alignment import get_stim_response
from modules.Alignment import get_stim_response_mode
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(labels)
        
        self.l_frames = 30
        self.r_frames = 50
        self.labels = labels
        [self.neu_seq, self.neu_time,
         self.stim_seq, self.trial_types,
         self.trial_isi_pre, self.trial_isi_post,
         self.outcome, self.stim_idx] = get_stim_response(
               neural_trials, self.l_frames, self.r_frames)
        [self.neu_seq_reward,
         self.neu_time_reward,
         self.reward_seq,
         self.lick_direc_reward] = get_outcome_response(
             neural_trials, 'trial_reward',
             self.l_frames, self.r_frames)
        [self.neu_seq_punish,
         self.neu_time_punish,
         self.punish_seq,
         self.lick_direc_punish] = get_outcome_response(
             neural_trials, 'trial_punish',
             self.l_frames, self.r_frames)
        self.significance = significance

    def plot_all(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        mean_neu = np.mean(neu_cate.reshape(
            -1, neu_cate.shape[2]), axis=0)
        sem_neu = sem(neu_cate.reshape(
            -1, neu_cate.shape[2]), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_neu, sem_neu, color, 'dff')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_onset(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'onset')
        mean_neu = np.mean(neu_cate.reshape(
            -1, neu_cate.shape[2]), axis=0)
        sem_neu = sem(neu_cate.reshape(
            -1, neu_cate.shape[2]), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        ax.fill_between(
            self.stim_seq + 1*(np.mean(self.trial_isi_pre)+self.stim_seq[1]),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        ax.fill_between(
            self.stim_seq + 2*(np.mean(self.trial_isi_pre)+self.stim_seq[1]),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid')
        self.plot_mean_sem(ax, self.neu_time, mean_neu, sem_neu, color, 'dff')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim onset (ms)')

    def plot_pre(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'pre_all')
        mean_neu = np.mean(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        sem_neu = sem(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_neu, sem_neu, color, 'dff')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre stim (ms)')

    def plot_pert(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*s,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate_post, trial_types_post, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_post, self.stim_idx, 'post_first')
        neu_short  = neu_cate_post[trial_types_post==1,:,:].reshape(-1, neu_cate.shape[2])
        mean_short = np.mean(neu_short, axis=0)
        sem_short  = sem(neu_short, axis=0)
        neu_long  = neu_cate_post[trial_types_post==2,:,:].reshape(-1, neu_cate.shape[2])
        mean_long = np.mean(neu_long, axis=0)
        sem_long = sem(neu_long, axis=0)
        upper = np.max([mean_long, mean_short]) + \
                np.max([sem_long, sem_short])
        lower = np.min([mean_long, mean_short]) - \
                np.max([sem_long, sem_short])
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short,
                      'red', 'short ISI')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long,
                      'blue', 'long ISI')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_post_isi(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*s,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, trial_isi_post = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_post, self.stim_idx, 'post_all')
        bins = [0, 205, 300, 400, 500, 600, 700, 800, 1000]
        isi_labels = ['100-200', '200-300', '300-400', '400-500',
                      '500-600', '600-700', '700=800', '800-900']
        bin_idx = np.digitize(trial_isi_post, bins)-1
        colors = plt.cm.gist_rainbow(np.arange(len(bins))/len(bins))
        m = []
        s = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                neu_post = neu_cate[bin_idx==i,:,:].reshape(-1, neu_cate.shape[2])
                neu_mean = np.mean(neu_post, axis=0)
                neu_sem = sem(neu_post, axis=0)
                self.plot_mean_sem(
                    ax, self.neu_time, neu_mean, neu_sem,
                    colors[i], isi_labels[i])
                m.append(neu_mean)
                s.append(neu_sem)
        upper = np.max(m) + np.max(s)
        lower = np.min(m) - np.max(s)
        ax.fill_between(
            self.stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        adjust_layout_neu(ax)
        ax.set_xlim([np.min(self.neu_time), np.max(self.neu_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])

    def plot_outcome(
            self, ax,
            neu_seq, neu_time, outcome, lick_direc,
            s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        neu_seq_left = neu_cate[lick_direc==0, :, :].copy()
        neu_seq_left = neu_seq_left.reshape(-1, neu_cate.shape[2])
        neu_seq_right = neu_cate[lick_direc==1, :, :].copy()
        neu_seq_right = neu_seq_right.reshape(-1, neu_cate.shape[2])
        mean_left  = np.mean(neu_seq_left, axis=0)
        mean_right = np.mean(neu_seq_right, axis=0)
        sem_left   = sem(neu_seq_left, axis=0)
        sem_right  = sem(neu_seq_right, axis=0)
        upper = np.max([np.max(mean_left), np.max(mean_right)]) + \
                np.max([np.max(sem_left), np.max(sem_right)])
        lower = np.min([np.min(mean_left), np.min(mean_right)]) - \
                np.max([np.max(sem_left), np.max(sem_right)])
        ax.fill_between(
            outcome,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='outcome')
        self.plot_mean_sem(ax, neu_time, mean_left, sem_left, 'mediumseagreen', 'left')
        self.plot_mean_sem(ax, neu_time, mean_right, sem_right, 'coral', 'right')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    # roi response to all stimulus.
    def roi_all(self, ax, roi_id):
        self.plot_all(ax, None, roi_id=roi_id)
        ax.set_title('response to all stim (all trials)')
    
    # roi response to all stimulus quantification.
    def roi_all_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time, color, 0, 0)
        ax.set_title('response to all stim (all trials)')
    
    # roi mean response to all stimulus single trial heatmap.
    def roi_all_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time, cmap, norm=True)
        ax.set_title('response to all stim (all trials)')
    
    # roi response to onset stimulus.
    def roi_onset(self, ax, roi_id):
        self.plot_onset(ax, None, roi_id=roi_id)
        ax.set_title('response to all stim (all trials)')
    
    # roi response to onset stimulus quantification.
    def roi_onset_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'onset')
        self.plot_win_mag_box(ax, neu_cate, self.neu_time, color, 0, 0)
        ax.set_title('response to onset stim (all trials)')
    
    # roi mean response to onset stimulus single trial heatmap.
    def roi_onset_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'onset')
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time, cmap, norm=True)
        ax.set_title('response to onset stim (all trials)')

    # roi response to pre pert stimulus.
    def roi_pre(self, ax, roi_id):
        self.plot_pre(ax, None, roi_id=roi_id)
        ax.set_title('response to pre pert stim (all trials)')
    
    # roi response to pre pert stimulus quantification.
    def roi_pre_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'pre_all')
        self.plot_win_mag_box(ax, neu_cate, self.neu_time, color, 0, 0)
        ax.set_title('response to pre pert stim (all trials)')
    
    # roi mean response to pre pert stimulus single trial heatmap.
    def roi_pre_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'pre_all')
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time, cmap, norm=True)
        ax.set_title('response to pre pert stim (all trials)')
        
    # roi response to perturbation stimulus.
    def roi_pert(self, ax, roi_id):
        self.plot_pert(ax, None, roi_id=roi_id)
        ax.set_title('response to 1st post pert stim (all trials)')
    
    # roi response to perturbation stimulus quantification.
    def roi_pert_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate_post, trial_types_post, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_post, self.stim_idx, 'post_first')
        neu_short = neu_cate_post[trial_types_post==1,:,:].copy()
        neu_long  = neu_cate_post[trial_types_post==2,:,:].copy()
        self.plot_win_mag_box(ax, neu_short, self.neu_time, 'red', 0, -0.1)
        self.plot_win_mag_box(ax, neu_long, self.neu_time, 'blue', 0, 0.1)
        ax.plot([], color='red', label='short ISI')
        ax.plot([], color='blue', label='long ISI')
        ax.legend(loc='upper right')
        ax.set_title('response to 1st post pert stim (all trials)')
    
    # roi mean response to perturbation stimulus single trial heatmap.
    def roi_pert_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_cate, _, _ = get_stim_response_mode(
            neu_cate, self.trial_types, self.trial_isi_pre, self.stim_idx, 'post_first')
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time, cmap, norm=True)
        ax.set_title('response to 1st post pert stim (all trials)')
    
    # roi response to post perturbation stimulus.
    def roi_post_isi(self, ax, roi_id):
        self.plot_post_isi(ax, None, roi_id=roi_id)
        ax.set_title('response to post pert stim')
    
    # roi response to reward.
    def roi_reward(self, ax, roi_id):
        self.plot_outcome(
            ax,
            self.neu_seq_reward, self.neu_time_reward,
            self.reward_seq, self.lick_direc_reward,
            None, roi_id=roi_id)
        ax.set_title('response to reward')
    
    # roi response to reward quantification.
    def roi_reward_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_reward[:,roi_id,:], axis=1)
        neu_left  = neu_cate[self.lick_direc_reward==0,:,:].copy()
        neu_right = neu_cate[self.lick_direc_reward==1,:,:].copy()
        self.plot_win_mag_box(
            ax, neu_left, self.neu_time_reward, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(
            ax, neu_right, self.neu_time_reward, 'coral', 0, 0.1)
        ax.plot([], color='mediumseagreen', label='left')
        ax.plot([], color='coral', label='right')
        ax.legend(loc='upper right')
        ax.set_title('response to reward')
    
    # roi mean response to reward single trial heatmap.
    def roi_reward_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq_reward[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time_reward, cmap, norm=True)
        ax.set_title('response to reward')
    
    # roi response to punish.
    def roi_punish(self, ax, roi_id):
        self.plot_outcome(
            ax,
            self.neu_seq_punish, self.neu_time_punish,
            self.punish_seq, self.lick_direc_punish,
            None, roi_id=roi_id)
        ax.set_title('response to punish')
    
    # roi response to punish quantification.
    def roi_punish_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
        neu_left  = neu_cate[self.lick_direc_punish==0,:,:].copy()
        neu_right = neu_cate[self.lick_direc_punish==1,:,:].copy()
        self.plot_win_mag_box(
            ax, neu_left, self.neu_time_punish, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(
            ax, neu_right, self.neu_time_punish, 'coral', 0, 0.1)
        ax.plot([], color='mediumseagreen', label='left')
        ax.plot([], color='coral', label='right')
        ax.legend(loc='upper right')
        ax.set_title('response to punish')
    
    # roi mean response to punish single trial heatmap.
    def roi_punish_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time_punish, cmap, norm=True)
        ax.set_title('response to punish')
        

class plotter_VIPTD_G8_align_perc(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    # excitory response to all stimulus.
    def all_exc(self, ax):
        self.plot_all(ax, self.significance['r_stim_all'], cate=-1)
        ax.set_title('excitory response to all stim (all trials)')

    # inhibitory response to all stimulus.
    def all_inh(self, ax):
        self.plot_all(ax, self.significance['r_stim_all'], cate=1)
        ax.set_title('inhibitory response to all stim (all trials)')

    # response to all stimulus heatmap average across trials.
    def all_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq, self.neu_time)
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to all stim (all trials)')

    # excitory response to stimulus onset.
    def onset_exc(self, ax):
        self.plot_onset(ax, self.significance['r_stim_onset'], cate=-1)
        ax.set_title('excitory response to stim onset (all trials)')

    # inhibitory response to stimulus onset.
    def onset_inh(self, ax):
        self.plot_onset(ax, self.significance['r_stim_onset'], cate=1)
        ax.set_title('inhibitory response to stim onset (all trials)')

    # response to stimulus onset heatmap average across trials.
    def onset_heatmap_neuron(self, ax):
        neu_seq_onset, _, _ = get_stim_response_mode(
            self.neu_seq, self.trial_types, self.trial_isi_pre, self.stim_idx, 'onset')
        self.plot_heatmap_neuron(ax, neu_seq_onset, self.neu_time)
        ax.set_xlabel('time since stim onset (ms)')
        ax.set_title('response to stim onset (all trials)')

    # excitory response to pre perturbation stimulus.
    def pre_exc(self, ax):
        self.plot_pre(ax, self.significance['r_stim_pre'], cate=-1)
        ax.set_title('excitory response to pre pert stim (all trials)')

    # inhibitory response to perturbation stimulus.
    def pre_inh(self, ax):
        self.plot_pre(ax, self.significance['r_stim_pre'], cate=1)
        ax.set_xlabel('time since 2nd pre stim (ms)')
        ax.set_title('inhibitory response to pre pert stim (all trials)')

    # response to perturbation stimulus heatmap average across trials.
    def pre_heatmap_neuron(self, ax):
        neu_seq_pre, _, _ = get_stim_response_mode(
            self.neu_seq, self.trial_types, self.trial_isi_pre, self.stim_idx, 'pre_all')
        self.plot_heatmap_neuron(ax, neu_seq_pre, self.neu_time)
        ax.set_xlabel('time since 2nd pre stim (ms)')
        ax.set_title('response to pre pert stim (all trials)')

    # excitory response to perturbation stimulus.
    def pert_exc(self, ax):
        self.plot_pert(ax, self.significance['r_stim_post_first'], cate=-1)
        ax.set_title('excitory response to 1st post pert stim (all trials)')

    # inhibitory response to perturbation stimulus.
    def pert_inh(self, ax):
        self.plot_pert(ax, self.significance['r_stim_post_first'], cate=1)
        ax.set_title('inhibitory response to 1st post pert stim (all trials)')

    # excitory response to post perturbation stimulus with isi.
    def post_isi_exc(self, ax):
        self.plot_post_isi(ax, self.significance['r_stim_post_all'], cate=-1)
        ax.set_title('excitory response to post pert stim')

    # inhibitory response to post perturbation stimulus with isi.
    def post_isi_inh(self, ax):
        self.plot_post_isi(ax, self.significance['r_stim_post_all'], cate=1)
        ax.set_title('inhibitory response to post pert stim')

    # excitory mean response to reward.
    def reward_exc(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward,
            self.reward_seq, self.lick_direc_reward,
            self.significance['r_reward'], cate=-1)
        ax.set_xlabel('time since reward start (ms)')
        ax.set_title('excitory response to reward')

    # inhibitory mean response to reward.
    def reward_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward,
            self.reward_seq, self.lick_direc_reward, 
            self.significance['r_reward'], cate=1)
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
            self.punish_seq, self.lick_direc_punish, 
            self.significance['r_punish'], cate=-1)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('excitory response to punish')

    # inhibitory mean response to punish.
    def punish_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish,
            self.punish_seq, self.lick_direc_punish,
            self.significance['r_punish'], cate=1)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('inhibitory response to punish')

    # response to punish heatmap average across trials.
    def punish_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_punish, self.neu_time_punish)
        ax.set_xlabel('time since punish start (ms)')
        ax.set_title('response to punish')
