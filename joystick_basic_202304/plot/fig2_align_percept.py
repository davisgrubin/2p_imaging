#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_grand
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance):
        super().__init__(labels)
        
        self.l_frames_vis1 = 5
        self.r_frames_vis1 = 75
        self.l_frames_vis2 = 30
        self.r_frames_vis2 = 50
        self.l_frames_out = 30
        self.r_frames_out = 50
        [self.neu_seq_vis1, self.neu_time_vis1,
         self.outcome_vis1, self.stim_seq_vis1] = get_stim_response(
                neural_trials, 'trial_vis1', self.l_frames_vis1, self.r_frames_vis1)
        [self.neu_seq_vis2, self.neu_time_vis2,
         self.outcome_vis2, self.stim_seq_vis2] = get_stim_response(
                neural_trials, 'trial_vis2', self.l_frames_vis2, self.r_frames_vis2)
        [self.neu_seq_reward, self.neu_time_reward,
         self.outcome_reward] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames_out, self.r_frames_out)
        [self.neu_seq_punish, self.neu_time_punish,
         self.outcome_punish] = get_outcome_response(
                neural_trials, 'trial_punish', self.l_frames_out, self.r_frames_out)
        self.significance = significance
             
    def plot_stim_outcome(
            self, ax,
            neu_seq, neu_time, outcome, stim_seq,
            s, 
            cate=None, roi_id=None):
        if len(neu_seq) > 0:
            if cate != None:
                neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            mean_all = np.mean(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
            mean_reward = np.mean(neu_cate[outcome==1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
            mean_punish = np.mean(neu_cate[outcome==-1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
            sem_all = sem(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
            sem_reward = sem(neu_cate[outcome==1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
            sem_punish = sem(neu_cate[outcome==-1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
            self.plot_mean_sem(ax, neu_time, mean_all,    sem_all,    'grey',           'all')
            self.plot_mean_sem(ax, neu_time, mean_reward, sem_reward, 'mediumseagreen', 'reward')
            self.plot_mean_sem(ax, neu_time, mean_punish, sem_punish, 'coral',          'punish')
            upper = np.max([mean_all, mean_reward, mean_punish]) +\
                    np.max([sem_all, sem_reward, sem_punish])
            lower = np.min([mean_all, mean_reward, mean_punish]) -\
                    np.max([sem_all, sem_reward, sem_punish])
            adjust_layout_grand(ax)
            ax.fill_between(
                stim_seq,
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='gold', alpha=0.15, step='mid', label='stim')
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    def plot_outcome(
            self, ax,
            neu_seq, neu_time, outcome, s,
            cate=None, roi_id=None):
        if cate != None:
            neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        mean_neu = np.mean(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        sem_neu = sem(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        self.plot_mean_sem(ax, neu_time, mean_neu, sem_neu, color, 'dff')
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.fill_between(
            outcome,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='outcome')
        adjust_layout_grand(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since outcome (ms)')
    
    # roi mean response to 1st stimulus with outcome.
    def roi_vis1_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # roi mean response to 1st stimulus with outcome quantification.
    def roi_vis1_outcome_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_vis1[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_vis1==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_vis1==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_vis1, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_vis1, 'coral', 0, 0.1)
        ax.set_title('response to 1st vis stim')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')
    
    # roi mean response to 1st stimulus with outcome single trial heatmap.
    def roi_vis1_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis1[:,roi_id,:], self.neu_time_vis1, cmap, norm=True)
        ax.set_title('response to 1st vis stim')
        
    # roi mean response to 2nd stimulus with outcome.
    def roi_vis2_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            None, roi_id=roi_id)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
    # roi mean response to 2nd stimulus with outcome quantification.
    def roi_vis2_outcome_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_vis2[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_vis2==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_vis2==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_vis2, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_vis2, 'coral', 0, 0.1)
        ax.set_title('response to 2nd vis stim')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')

    # roi mean response to 2nd stimulus with outcome single trial heatmap.
    def roi_vis2_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis2[:,roi_id,:], self.neu_time_vis2, cmap, norm=True)
        ax.set_title('response to 2nd vis stim')
    
    # roi mean response to reward.
    def roi_reward(self, ax, roi_id):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_reward, None, roi_id=roi_id)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # roi mean response to reward quantification.
    def roi_reward_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_reward[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time_reward, color, 0, 0)
        ax.set_title('response to reward')

    # roi mean response to reward single trial heatmap.
    def roi_reward_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_reward[:,roi_id,:], self.neu_time_reward, cmap, norm=True)
        ax.set_title('response to reward')
    
    # roi mean response to punish.
    def roi_punish(self, ax, roi_id):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish, 
            self.outcome_punish, None, roi_id=roi_id)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to punish')

    # roi mean response to punish quantification.
    def roi_punish_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time_punish, color, 0, 0)
        ax.set_title('response to punish')
    
    # roi mean response to punish single trial heatmap.
    def roi_punish_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_punish[:,roi_id,:], self.neu_time_punish, cmap, norm=True)
        ax.set_title('response to punish')
        
class plotter_VIPTD_G8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    # excitory mean response to 1st stimulus with outcome.
    def vis1_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.significance['r_vis1'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim')
    
    # inhibitory mean response to 1st stimulus with outcome.
    def vis1_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.significance['r_vis1'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials.
    def vis1_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_vis1, self.neu_time_vis1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # excitory mean response to 2nd stimulus with outcome.
    def vis2_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2, 
            self.significance['r_vis2'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim')
    
    # inhibitory mean response to 2nd stimulus with outcome.
    def vis2_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2, 
            self.significance['r_vis2'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials.
    def vis2_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_vis2, self.neu_time_vis2)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
    # excitory mean response to reward.
    def reward_exc(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_reward, self.significance['r_reward'], cate=-1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('excitory response to reward')
    
    # inhibitory mean response to reward.
    def reward_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_reward, self.significance['r_reward'], cate=1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('inhibitory response to reward')
    
    # response to reward heatmap average across trials.
    def reward_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_reward, self.neu_time_reward)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # excitory mean response to punish.
    def punish_exc(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish, 
            self.outcome_punish, self.significance['r_punish'], cate=-1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('excitory response to punish')
    
    # inhibitory mean response to punish.
    def punish_inh(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish, 
            self.outcome_punish, self.significance['r_punish'], cate=1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('inhibitory response to punish')
    
    # response to punish heatmap average across trials.
    def punish_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_punish, self.neu_time_punish)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to punish')
        
        
class plotter_VIPG8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    # mean response to 1st stimulus with outcome.
    def vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            cate=1)
        ax.set_title('response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials.
    def vis1_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_vis1, self.neu_time_vis1)
        ax.set_title('response to 1st vis stim')
    
    # mean response to 2nd stimulus with outcome.
    def vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2, 
            cate=1)
        ax.set_title('response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials.
    def vis2_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_vis2, self.neu_time_vis2)
        ax.set_title('response to 2nd vis stim')
    
    # mean response to reward.
    def reward(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_reward, cate=1)
        ax.set_title('response to reward')
    
    # response to reward heatmap average across trials.
    def reward_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_reward, self.neu_time_reward)
        ax.set_title('response to reward')
    
    # mean response to punish.
    def punish(self, ax):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish, 
            self.outcome_punish, cate=1)
        ax.set_title('response to punish')
    
    # response to punish heatmap average across trials.
    def punish_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_punish, self.neu_time_punish)
        ax.set_title('response to punish')