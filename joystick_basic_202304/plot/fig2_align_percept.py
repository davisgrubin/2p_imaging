#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from plot.utils import get_block_epoch
from plot.utils import get_trial_type
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(labels)
        self.l_frames_vis1 = 5
        self.r_frames_vis1 = 75
        self.l_frames_vis2 = 30
        self.r_frames_vis2 = 50
        self.l_frames_out = 30
        self.r_frames_out = 50
        [self.neu_seq_vis1, self.neu_time_vis1,
         self.outcome_vis1, self.stim_seq_vis1,
         self.delay_vis1] = get_stim_response(
                neural_trials, 'trial_vis1', self.l_frames_vis1, self.r_frames_vis1)
        [self.neu_seq_vis2, self.neu_time_vis2,
         self.outcome_vis2, self.stim_seq_vis2,
         self.delay_vis2] = get_stim_response(
                neural_trials, 'trial_vis2', self.l_frames_vis2, self.r_frames_vis2)
        [self.neu_seq_reward, self.neu_time_reward,
         self.outcome_seq_reward, self.outcome_reward,
         self.delay_reward] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames_out, self.r_frames_out)
        [self.neu_seq_punish, self.neu_time_punish,
         self.outcome_seq_punish, self.outcome_punish,
         self.delay_punish] = get_outcome_response(
                neural_trials, 'trial_punish', self.l_frames_out, self.r_frames_out)
        self.significance = significance
        self.cate_delay = cate_delay
             
    def plot_stim_outcome(
            self, ax,
            neu_seq, neu_time, outcome, stim_seq,
            delay, block, s, cate=None, roi_id=None):
        if not np.isnan(np.sum(neu_seq)):
            if cate != None:
                neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            idx = get_trial_type(self.cate_delay, delay, block)
            mean = []
            sem = []
            for i in range(4):
                trial_idx = idx*(outcome==i)
                if len(trial_idx) >= self.min_num_trial:
                    m, s = get_mean_sem(neu_cate[trial_idx,:,:])
                    self.plot_mean_sem(ax, neu_time, m, s, self.colors[i], self.states[i])
                    mean.append(m)
                    sem.append(s)
            upper = np.nanmax(mean) + np.nanmax(sem)
            lower = np.nanmin(mean) - np.nanmax(sem)
            adjust_layout_neu(ax)
            ax.fill_between(
                stim_seq,
                lower, upper,
                color='grey', alpha=0.15, step='mid', label='stim')
            ax.set_ylim([lower, upper])
    
    def plot_reward(
            self, ax,
            delay, block, s,
            cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_reward[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_reward[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        idx = get_trial_type(self.cate_delay, delay, block)
        neu_mean, neu_sem = get_mean_sem(neu_cate[idx,:,:])
        self.plot_mean_sem(ax, self.neu_time_reward, neu_mean, neu_sem, color, 'dff')
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        ax.fill_between(
            self.outcome_seq_reward,
            lower, upper,
            color='grey', alpha=0.15, step='mid')
        adjust_layout_neu(ax)
        ax.set_ylim([lower, upper])
        ax.set_xlabel('time since reward (ms)')
    
    def plot_punish(
            self, ax,
            block, s,
            cate=None, roi_id=None):
        if not np.isnan(np.sum(self.neu_seq_punish)):
            if cate != None:
                neu_cate = self.neu_seq_punish[:,(self.labels==cate)*s,:]
            if roi_id != None:
                neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
            idx = get_trial_type(self.cate_delay, self.delay_punish, block)
            mean = []
            sem = []
            for i in [1,2,3]:
                trial_idx = idx*(self.outcome_punish==i)
                if len(trial_idx) >= self.min_num_trial:
                    m, s = get_mean_sem(neu_cate[trial_idx,:,:])
                    self.plot_mean_sem(ax, self.neu_time_punish, m, s, self.colors[i], self.states[i])
                    mean.append(m)
                    sem.append(s)
            upper = np.nanmax(mean) + np.nanmax(sem)
            lower = np.nanmin(mean) - np.nanmax(sem)
        ax.fill_between(
            self.outcome_seq_punish,
            lower, upper,
            color='grey', alpha=0.15, step='mid')
        adjust_layout_neu(ax)
        ax.set_ylim([lower, upper])
        ax.set_xlabel('time since punish (ms)')
    
    def plot_stim_epoch(
            self, ax,
            neu_seq, neu_time, outcome, stim_seq,
            delay, block, s,
            cate=None, roi_id=None):
        if not np.isnan(np.sum(neu_seq)):
            if cate != None:
                neu_cate = neu_seq[:,(self.labels==cate)*s,:]
                _, color1, color2, _ = get_roi_label_color([cate], 0)
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
                _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            idx = get_trial_type(self.cate_delay, delay, block)
            trial_idx, block_tran = get_block_epoch(idx)
            if np.sum(outcome==0) != 0:
                i_ep1 = (block_tran==1) * trial_idx * idx * (outcome==0)
                i_ep2 = (block_tran==0) * trial_idx * idx * (outcome==0)
            else:
                i_ep1 = (block_tran==1) * trial_idx * idx * (outcome>0)
                i_ep2 = (block_tran==0) * trial_idx * idx * (outcome>0)
            m_ep1, s_ep1 = get_mean_sem(neu_cate[i_ep1,:,:])
            m_ep2, s_ep2 = get_mean_sem(neu_cate[i_ep2,:,:])
            if not np.isnan(np.sum(m_ep1)) and not np.isnan(np.sum(m_ep2)):
                self.plot_mean_sem(ax, neu_time, m_ep1, s_ep1, color1, 'ep1')
                self.plot_mean_sem(ax, neu_time, m_ep2, s_ep2, color2, 'ep2')
                upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
                lower = np.nanmin([m_ep1, m_ep2]) - np.nanmax([s_ep1, s_ep2])
                ax.fill_between(
                    stim_seq,
                    lower, upper,
                    color='grey', alpha=0.15, step='mid')
                adjust_layout_neu(ax)
                ax.set_ylim([lower, upper])
    
    # roi response to Vis1 with outcome.
    def roi_vis1_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1')
    
    # roi response to Vis1 with outcome quantification.
    def roi_vis1_outcome_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_vis1[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_vis1==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_vis1==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_vis1, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_vis1, 'coral', 0, 0.1)
        ax.set_title('response to Vis1')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')
    
    # roi response to Vis1 with outcome single trial heatmap.
    def roi_vis1_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis1[:,roi_id,:], self.neu_time_vis1, cmap, norm=True)
        ax.set_title('response to Vis1')
        
    # roi response to Vis2 with outcome.
    def roi_vis2_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2')
    
    # roi response to Vis2 with outcome quantification.
    def roi_vis2_outcome_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_vis2[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_vis2==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_vis2==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_vis2, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_vis2, 'coral', 0, 0.1)
        ax.set_title('response to Vis2')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')

    # roi response to Vis2 with outcome single trial heatmap.
    def roi_vis2_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis2[:,roi_id,:], self.neu_time_vis2, cmap, norm=True)
        ax.set_title('response to Vis2')
    
    # roi response to reward.
    def roi_reward(self, ax, roi_id):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_seq_reward, None, roi_id=roi_id)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # roi response to reward quantification.
    def roi_reward_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_reward[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time_reward, color, 0, 0)
        ax.set_title('response to reward')

    # roi response to reward single trial heatmap.
    def roi_reward_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_reward[:,roi_id,:], self.neu_time_reward, cmap, norm=True)
        ax.set_title('response to reward')
    
    # roi response to punish.
    def roi_punish(self, ax, roi_id):
        self.plot_outcome(
            ax, self.neu_seq_punish, self.neu_time_punish, 
            self.outcome_seq_punish, None, roi_id=roi_id)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to punish')

    # roi response to punish quantification.
    def roi_punish_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time_punish, color, 0, 0)
        ax.set_title('response to punish')
    
    # roi response to punish single trial heatmap.
    def roi_punish_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_punish[:,roi_id,:], self.neu_time_punish, cmap, norm=True)
        ax.set_title('response to punish')
        
class plotter_VIPTD_G8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    def plot_exc_inh(self, ax, neu_seq, neu_time, stim_seq, delay, block, s):
        if not np.isnan(np.sum(neu_seq)):
            _, _, color_exc, _ = get_roi_label_color([-1], 0)
            _, _, color_inh, _ = get_roi_label_color([1], 0)
            idx = get_trial_type(self.cate_delay, delay, block)
            neu_cate = neu_seq[idx,:,:].copy()
            mean_exc, sem_exc = get_mean_sem(neu_cate[:,(self.labels==-1)*s,:])
            mean_inh, sem_inh = get_mean_sem(neu_cate[:,(self.labels==1)*s,:])
            self.plot_mean_sem(ax, neu_time, mean_exc, sem_exc, color_exc, 'exc')
            self.plot_mean_sem(ax, neu_time, mean_inh, sem_inh, color_inh, 'inh')
            upper = np.nanmax([mean_exc, mean_inh]) + np.nanmax([sem_exc, sem_inh])
            lower = np.nanmin([mean_exc, mean_inh]) - np.nanmax([sem_exc, sem_inh])
            adjust_layout_neu(ax)
            ax.fill_between(
                stim_seq,
                lower, upper,
                color='grey', alpha=0.15, step='mid')
            ax.set_ylim([lower, upper])
            
    # excitory response to Vis1 with outcome (short).
    def short_vis1_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1')
    
    # inhibitory response to Vis1 with outcome (short).
    def short_vis1_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1')
    
    # response to Vis1 heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1')
    
    # excitory response to Vis2 with outcome (short).
    def short_vis2_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2')
    
    # inhibitory response to Vis2 with outcome (short).
    def short_vis2_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2')
    
    # response to Vis2 heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2')
    
    # excitory response to reward (short).
    def short_reward_exc(self, ax):
        self.plot_reward(ax, self.delay_reward, 0, self.significance['r_reward'], cate=-1)
        ax.set_title('excitory response to reward')
    
    # inhibitory response to reward (short).
    def short_reward_inh(self, ax):
        self.plot_reward(ax, self.delay_reward, 0, self.significance['r_reward'], cate=1)
        ax.set_title('inhibitory response to reward')
    
    # response to reward heatmap average across trials (short).
    def short_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # excitory response to punish (short).
    def short_punish_exc(self, ax):
        self.plot_punish(ax, 0, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # inhibitory response to punish (short).
    def short_punish_inh(self, ax):
        self.plot_punish(ax, 0, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to punish (all)')
    
    # excitory response to Vis1 with epoch (short).
    def short_epoch_vis1_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1 (reward)')
    
    # inhibitory response to Vis1 with epoch (short).
    def short_epoch_vis1_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (reward)')
    
    # excitory response to Vis2 with epoch (short).
    def short_epoch_vis2_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2 (reward)')
    
    # inhibitory response to Vis2 with epoch (short).
    def short_epoch_vis2_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (reward)')
    
    # excitory response to reward with epoch (short).
    def short_epoch_reward_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_reward, self.neu_time_reward,
            self.outcome_reward, self.outcome_seq_reward,
            self.delay_reward, 0,
            self.significance['r_reward'], cate=-1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('excitory response to reward')
    
    # inhibitory response to reward with epoch (short).
    def short_epoch_reward_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_reward, self.neu_time_reward,
            self.outcome_reward, self.outcome_seq_reward,
            self.delay_reward, 0,
            self.significance['r_reward'], cate=1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('inhibitory response to reward')
    
    # excitory response to punish with epoch (short).
    def short_epoch_punish_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_punish, self.neu_time_punish,
            self.outcome_punish, self.outcome_seq_punish,
            self.delay_punish, 0,
            self.significance['r_punish'], cate=-1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('excitory response to punish (all)')
    
    # inhibitory response to punish with epoch (short).
    def short_epoch_punish_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_punish, self.neu_time_punish,
            self.outcome_punish, self.outcome_seq_punish,
            self.delay_punish, 0,
            self.significance['r_punish'], cate=1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('inhibitory response to punish (all)')
    
    # excitory response to Vis1 with outcome (long).
    def long_vis1_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1')
    
    # inhibitory response to Vis1 with outcome (long).
    def long_vis1_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1')
    
    # response to Vis1 heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1')
    
    # excitory response to Vis2 with outcome (long).
    def long_vis2_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2')
    
    # inhibitory response to Vis2 with outcome (long).
    def long_vis2_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2')
    
    # response to Vis2 heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2')
    
    # excitory response to reward (long).
    def long_reward_exc(self, ax):
        self.plot_reward(ax, self.delay_reward, 1, self.significance['r_reward'], cate=-1)
        ax.set_title('excitory response to reward')
    
    # inhibitory response to reward (long).
    def long_reward_inh(self, ax):
        self.plot_reward(ax, self.delay_reward, 1, self.significance['r_reward'], cate=1)
        ax.set_title('inhibitory response to reward')
    
    # response to reward heatmap average across trials (long).
    def long_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # excitory response to punish (long).
    def long_punish_exc(self, ax):
        self.plot_punish(ax, 1, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # inhibitory response to punish (long).
    def long_punish_inh(self, ax):
        self.plot_punish(ax, 1, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to Vis1 with epoch (long).
    def long_epoch_vis1_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1 (reward)')
    
    # inhibitory response to Vis1 with epoch (long).
    def long_epoch_vis1_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (reward)')
    
    # excitory response to Vis2 with epoch (long).
    def long_epoch_vis2_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2 (reward)')
    
    # inhibitory response to Vis2 with epoch (long).
    def long_epoch_vis2_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (reward)')
    
    # excitory response to reward with epoch (long).
    def long_epoch_reward_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_reward, self.neu_time_reward,
            self.outcome_reward, self.outcome_seq_reward,
            self.delay_reward, 1,
            self.significance['r_reward'], cate=-1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('excitory response to reward')
    
    # inhibitory response to reward with epoch (long).
    def long_epoch_reward_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_reward, self.neu_time_reward,
            self.outcome_reward, self.outcome_seq_reward,
            self.delay_reward, 1,
            self.significance['r_reward'], cate=1)
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('inhibitory response to reward')

    # excitory response to punish with epoch (long).
    def long_epoch_punish_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_punish, self.neu_time_punish,
            self.outcome_punish, self.outcome_seq_punish,
            self.delay_punish, 1,
            self.significance['r_punish'], cate=-1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('excitory response to punish')
    
    # inhibitory response to punish with epoch (long).
    def long_epoch_punish_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_punish, self.neu_time_punish,
            self.outcome_punish, self.outcome_seq_punish,
            self.delay_punish, 1,
            self.significance['r_punish'], cate=1)
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('inhibitory response to punish')
        
    # response to Vis1 (short).
    def short_exc_inh_vis1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1 (short)')
        
    # response to Vis2 (short).
    def short_exc_inh_vis2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2 (short)')
    
    # response to Vis1 (long).
    def long_exc_inh_vis1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1 (long)')
        
    # response to Vis2 (long).
    def long_exc_inh_vis2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2 (long)')
        
        
class plotter_L7G8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)

    # excitory response to Vis1 with outcome (short).
    def short_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1')
    
    # response to Vis1 heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1')
    
    # excitory response to Vis2 with outcome (short).
    def short_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2')
    
    # response to Vis2 heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2')
    
    # excitory response to reward (short).
    def short_reward(self, ax):
        self.plot_reward(ax, self.delay_reward, 0, self.significance['r_reward'], cate=-1)
        ax.set_title('excitory response to reward')
    
    # response to reward heatmap average across trials (short).
    def short_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # excitory response to punish (short).
    def short_punish(self, ax):
        self.plot_punish(ax, 0, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to Vis1 with epoch (short).
    def short_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1 (reward)')
    
    # excitory response to Vis2 with epoch (short).
    def short_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2 (reward)')
    
    # excitory response to Vis1 with outcome (long).
    def long_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1')
    
    # response to Vis1 heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1')
    
    # excitory response to Vis2 with outcome (long).
    def long_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2')
    
    # response to Vis2 heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2')
    
    # excitory response to reward (long).
    def long_reward(self, ax):
        self.plot_reward(ax, self.delay_reward, 1, self.significance['r_reward'], cate=-1)
        ax.set_title('excitory response to reward')
    
    # response to reward heatmap average across trials (long).
    def long_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward')
    
    # excitory response to punish (long).
    def long_punish(self, ax):
        self.plot_punish(ax, 1, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to Vis1 with epoch (long).
    def long_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('excitory response to Vis1 (reward)')
    
    # excitory response to Vis2 with epoch (long).
    def long_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('excitory response to Vis2 (reward)')


class plotter_VIPG8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    # inhibitory response to Vis1 with outcome (short).
    def short_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (short)')
    
    # response to Vis1 heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1 (short)')

    # inhibitory response to Vis2 with outcome (short).
    def short_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (short)')
    
    # response to Vis2 heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2 (short)')
    
    # inhibitory response to reward (short).
    def short_reward(self, ax):
        self.plot_reward(ax, self.delay_reward, 0, self.significance['r_reward'], cate=1)
        ax.set_title('inhibitory response to reward (short)')
    
    # response to reward heatmap average across trials (short).
    def short_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward (short)')

    # inhibitory response to punish (short).
    def short_punish(self, ax):
        self.plot_punish(ax, 0, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish (short)')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish (short)')
    
    # inhibitory response to Vis1 with epoch (short).
    def short_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (short) (reward)')
    
    # inhibitory response to Vis2 with epoch (short).
    def short_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (short) (reward)')
    
    # inhibitory response to Vis1 with outcome (long).
    def long_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (long)')
    
    # response to Vis1 heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('response to Vis1 (long)')
    
    # inhibitory response to Vis2 with outcome (long).
    def long_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (long)')
    
    # response to Vis2 heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('response to Vis2 (long)')
    
    # inhibitory response to reward (long).
    def long_reward(self, ax):
        self.plot_reward(ax, self.delay_reward, 1, self.significance['r_reward'], cate=1)
        ax.set_title('inhibitory response to reward (long)')
    
    # response to reward heatmap average across trials (long).
    def long_reward_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_reward, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_reward[idx,:,:], self.neu_time_reward, self.significance['r_reward'])
        ax.set_xlabel('time since reward (ms)')
        ax.set_title('response to reward (long)')

    # inhibitory response to punish (long).
    def long_punish(self, ax):
        self.plot_punish(ax, 1, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish (long)')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish (long)')
    
    # inhibitory response to Vis1 with epoch (long).
    def long_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('inhibitory response to Vis1 (long) (reward)')
    
    # inhibitory response to Vis2 with epoch (long).
    def long_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('inhibitory response to Vis2 (long) (reward)')