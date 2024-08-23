#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from plot.utils import get_block_transient
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
         self.outcome_reward, _,
         self.delay_reward] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames_out, self.r_frames_out)
        [self.neu_seq_punish, self.neu_time_punish,
         self.outcome_punish, self.punish_label,
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
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='grey', alpha=0.15, step='mid', label='stim')
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
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
            self.outcome_reward,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='grey', alpha=0.15, step='mid')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since reward (ms)')
    
    def plot_punish(
            self, ax,
            delay, block, s,
            cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_punish[:,(self.labels==cate)*s,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_punish[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        idx = get_trial_type(self.cate_delay, delay, block)
        neu1 = neu_cate[idx*(self.punish_label=='no1stpush'),:,:].copy()
        neu2 = neu_cate[idx*(self.punish_label=='no2ndpush'),:,:].copy()
        m = []
        s = []
        if np.size(neu1)>0:
            neu1_mean, neu1_sem = get_mean_sem(neu1)
            self.plot_mean_sem(ax, self.neu_time_punish, neu1_mean, neu1_sem, color1, 'no1stpush')
            m.append(neu1_mean)
            s.append(neu1_sem)
        if np.size(neu2)>0:
            neu2_mean, neu2_sem = get_mean_sem(neu2)
            self.plot_mean_sem(ax, self.neu_time_punish, neu2_mean, neu2_sem, color2, 'no2ndpush')
            m.append(neu2_mean)
            s.append(neu2_sem)
        upper = np.nanmax(m) + np.nanmax(s)
        lower = np.nanmin(m) - np.nanmax(s)
        ax.fill_between(
            self.outcome_punish,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='grey', alpha=0.15, step='mid')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
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
            trial_idx, block_tran = get_block_transient(idx)
            i_ep1 = (block_tran==1) * trial_idx * idx * (outcome==0)
            i_ep2 = (block_tran==0) * trial_idx * idx * (outcome==0)
            m_ep1, s_ep1 = get_mean_sem(neu_cate[i_ep1,:,:])
            m_ep2, s_ep2 = get_mean_sem(neu_cate[i_ep2,:,:])
            if not np.isnan(np.sum(m_ep1)) and not np.isnan(np.sum(m_ep2)):
                self.plot_mean_sem(ax, neu_time, m_ep1, s_ep1, color1, 'ep1')
                self.plot_mean_sem(ax, neu_time, m_ep2, s_ep2, color2, 'ep2')
                upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
                lower = np.nanmin([m_ep1, m_ep2]) - np.nanmax([s_ep1, s_ep2])
                ax.fill_between(
                    stim_seq,
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color='grey', alpha=0.15, step='mid')
                adjust_layout_neu(ax)
                ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    # roi response to 1st stimulus with outcome.
    def roi_vis1_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # roi response to 1st stimulus with outcome quantification.
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
    
    # roi response to 1st stimulus with outcome single trial heatmap.
    def roi_vis1_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis1[:,roi_id,:], self.neu_time_vis1, cmap, norm=True)
        ax.set_title('response to 1st vis stim')
        
    # roi response to 2nd stimulus with outcome.
    def roi_vis2_outcome(self, ax, roi_id):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            None, roi_id=roi_id)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
    # roi response to 2nd stimulus with outcome quantification.
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

    # roi response to 2nd stimulus with outcome single trial heatmap.
    def roi_vis2_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_vis2[:,roi_id,:], self.neu_time_vis2, cmap, norm=True)
        ax.set_title('response to 2nd vis stim')
    
    # roi response to reward.
    def roi_reward(self, ax, roi_id):
        self.plot_outcome(
            ax, self.neu_seq_reward, self.neu_time_reward, 
            self.outcome_reward, None, roi_id=roi_id)
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
            self.outcome_punish, None, roi_id=roi_id)
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
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='grey', alpha=0.15, step='mid')
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            
    # excitory response to 1st stimulus with outcome (short).
    def short_vis1_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim')
    
    # inhibitory response to 1st stimulus with outcome (short).
    def short_vis1_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # excitory response to 2nd stimulus with outcome (short).
    def short_vis2_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim')
    
    # inhibitory response to 2nd stimulus with outcome (short).
    def short_vis2_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
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
        self.plot_punish(ax, self.delay_punish, 0, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # inhibitory response to punish (short).
    def short_punish_inh(self, ax):
        self.plot_punish(ax, self.delay_punish, 0, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to 1st stimulus with epoch (short).
    def short_epoch_vis1_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim (reward)')
    
    # inhibitory response to 1st stimulus with epoch (short).
    def short_epoch_vis1_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (reward)')
    
    # excitory response to 2nd stimulus with epoch (short).
    def short_epoch_vis2_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim (reward)')
    
    # inhibitory response to 2nd stimulus with epoch (short).
    def short_epoch_vis2_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (reward)')
    
    # excitory response to 1st stimulus with outcome (long).
    def long_vis1_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim')
    
    # inhibitory response to 1st stimulus with outcome (long).
    def long_vis1_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # excitory response to 2nd stimulus with outcome (long).
    def long_vis2_outcome_exc(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim')
    
    # inhibitory response to 2nd stimulus with outcome (long).
    def long_vis2_outcome_inh(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
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
        self.plot_punish(ax, self.delay_punish, 1, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # inhibitory response to punish (long).
    def long_punish_inh(self, ax):
        self.plot_punish(ax, self.delay_punish, 1, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to 1st stimulus with epoch (long).
    def long_epoch_vis1_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim (reward)')
    
    # inhibitory response to 1st stimulus with epoch (long).
    def long_epoch_vis1_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (reward)')
    
    # excitory response to 2nd stimulus with epoch (long).
    def long_epoch_vis2_exc(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim (reward)')
    
    # inhibitory response to 2nd stimulus with epoch (long).
    def long_epoch_vis2_inh(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (reward)')
    
    # response to 1st stimulus (short).
    def short_exc_inh_vis1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim (short)')
        
    # response to 2nd stimulus (short).
    def short_exc_inh_vis2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim (short)')
    
    # response to 1st stimulus (long).
    def long_exc_inh_vis1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim (long)')
        
    # response to 2nd stimulus (long).
    def long_exc_inh_vis2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim (long)')
        
        
class plotter_L7G8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)

    # excitory response to 1st stimulus with outcome (short).
    def short_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # excitory response to 2nd stimulus with outcome (short).
    def short_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
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
        self.plot_punish(ax, self.delay_punish, 0, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to 1st stimulus with epoch (short).
    def short_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim (reward)')
    
    # excitory response to 2nd stimulus with epoch (short).
    def short_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim (reward)')
    
    # excitory response to 1st stimulus with outcome (long).
    def long_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim')
    
    # response to 1st stimulus heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim')
    
    # excitory response to 2nd stimulus with outcome (long).
    def long_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim')
    
    # response to 2nd stimulus heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim')
    
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
        self.plot_punish(ax, self.delay_punish, 1, self.significance['r_punish'], cate=-1)
        ax.set_title('excitory response to punish')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish')
    
    # excitory response to 1st stimulus with epoch (long).
    def long_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('excitory response to 1st vis stim (reward)')
    
    # excitory response to 2nd stimulus with epoch (long).
    def long_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=-1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('excitory response to 2nd vis stim (reward)')


class plotter_VIPG8_percept(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    # inhibitory response to 1st stimulus with outcome (short).
    def short_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (short)')
    
    # response to 1st stimulus heatmap average across trials (short).
    def short_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim (short)')

    # inhibitory response to 2nd stimulus with outcome (short).
    def short_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (short)')
    
    # response to 2nd stimulus heatmap average across trials (short).
    def short_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim (short)')
    
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
        self.plot_punish(ax, self.delay_punish, 0, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish (short)')
    
    # response to punish heatmap average across trials (short).
    def short_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish (short)')
    
    # inhibitory response to 1st stimulus with epoch (short).
    def short_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (short) (reward)')
    
    # inhibitory response to 2nd stimulus with epoch (short).
    def short_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 0,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (short) (reward)')
    
    # inhibitory response to 1st stimulus with outcome (long).
    def long_vis1_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (long)')
    
    # response to 1st stimulus heatmap average across trials (long).
    def long_vis1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis1[idx,:,:], self.neu_time_vis1, self.significance['r_vis'])
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('response to 1st vis stim (long)')
    
    # inhibitory response to 2nd stimulus with outcome (long).
    def long_vis2_outcome(self, ax):
        self.plot_stim_outcome(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (long)')
    
    # response to 2nd stimulus heatmap average across trials (long).
    def long_vis2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_vis2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_vis2[idx,:,:], self.neu_time_vis2, self.significance['r_vis'])
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('response to 2nd vis stim (long)')
    
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
        self.plot_punish(ax, self.delay_punish, 1, self.significance['r_punish'], cate=1)
        ax.set_title('inhibitory response to punish (long)')
    
    # response to punish heatmap average across trials (long).
    def long_punish_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_punish, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_punish[idx,:,:], self.neu_time_punish, self.significance['r_punish'])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('response to all punish (long)')
    
    # inhibitory response to 1st stimulus with epoch (long).
    def long_epoch_vis1(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis1, self.neu_time_vis1,
            self.outcome_vis1, self.stim_seq_vis1,
            self.delay_vis1, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 1st stim (ms)')
        ax.set_title('inhibitory response to 1st vis stim (long) (reward)')
    
    # inhibitory response to 2nd stimulus with epoch (long).
    def long_epoch_vis2(self, ax):
        self.plot_stim_epoch(
            ax,
            self.neu_seq_vis2, self.neu_time_vis2,
            self.outcome_vis2, self.stim_seq_vis2,
            self.delay_vis2, 1,
            self.significance['r_vis'], cate=1)
        ax.set_xlabel('time since 2nd stim (ms)')
        ax.set_title('inhibitory response to 2nd vis stim (long) (reward)')