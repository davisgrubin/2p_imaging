#!/usr/bin/env python3

import numpy as np
from modules.Alignment import get_js_pos
from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from modules.Alignment import get_motor_response
from plot.utils import get_mean_sem
from plot.utils import get_trial_outcome
from plot.utils import get_trial_type
from plot.utils import adjust_layout_js
from plot.utils import utils


class plotter_all_beh(utils):
    
    def __init__(self, neural_trials, cate_delay):
        super().__init__(None)
        self.neural_trials = neural_trials
        self.outcomes = get_trial_outcome(neural_trials)
        self.l_frames_vis1 = 5
        self.r_frames_vis1 = 75
        self.l_frames_vis2 = 30
        self.r_frames_vis2 = 50
        self.l_frames_out = 30
        self.r_frames_out = 50
        self.l_frames_motor = 10
        self.r_frames_motor = 70
        [_, self.neu_time_vis1, _, _, _] = get_stim_response(
                neural_trials, 'trial_vis1', self.l_frames_vis1, self.r_frames_vis1)
        [_, self.neu_time_vis2, _, _, _] = get_stim_response(
                neural_trials, 'trial_vis2', self.l_frames_vis2, self.r_frames_vis2)
        [_, self.neu_time_out, _, _, _] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames_out, self.r_frames_out)
        [_, self.neu_time_motor, _, _] = get_motor_response(
            neural_trials, 'trial_push1', self.l_frames_motor, self.r_frames_motor)
        [_, self.neu_time_wait2, _, _] = get_motor_response(
            neural_trials, 'trial_wait2', self.l_frames_motor, self.r_frames_motor)
        self.cate_delay = cate_delay

    def plot_align_pos_outcome(
            self, ax,
            align_data, align_time,
            outcome, idx,
            neu_time):
        l_idx = np.argmin(np.abs(align_time - neu_time[0]))
        r_idx = np.argmin(np.abs(align_time - neu_time[-1]))
        align_time = align_time[l_idx:r_idx]
        mean = []
        sem = []
        for i in range(4):
            trial_idx = idx*(outcome==i)
            if len(trial_idx) >= self.min_num_trial:
                m, s = get_mean_sem(align_data[trial_idx,:])
                m = m[l_idx:r_idx]
                s = s[l_idx:r_idx]
                self.plot_mean_sem(ax, align_time, m, s, self.colors[i], self.states[i])
                mean.append(m)
                sem.append(s)
        upper = np.nanmax(mean) + np.nanmax(sem)
        lower = -0.1
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(neu_time), np.nanmax(neu_time)])
        ax.set_ylim([lower, 1.1*upper])
        
    # outcome percentage.
    def session_outcome(self, ax):
        trial_delay = np.array([self.neural_trials[t]['trial_delay']
                  for t in self.neural_trials.keys()])
        trial_types = get_trial_type(self.cate_delay, trial_delay, 1).astype('int32')
        for block in [0,1]:
            bottom = 0
            for i in range(4):
                pc = np.sum((trial_types==block) * (self.outcomes==i)) / np.sum(trial_types==block)
                ax.bar(
                    block, pc,
                    bottom=bottom, edgecolor='white', width=0.25, color=self.colors[i])
                bottom += pc
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True)
        ax.set_xlabel('trial type')
        ax.set_ylabel('percentage')
        ax.set_xlim([-1,3])
        ax.set_ylim([0,1])
        ax.set_xticks([0,1])
        ax.set_xticklabels(['short', 'long'])
        for i in range(len(self.states)):
            ax.plot([], color=self.colors[i], label=self.states[i])
        ax.legend(loc='upper right')
        ax.set_title('percentage of outcome for {} trials'.format(
            len(self.outcomes)))
    
    # delay curve.
    def delay_dist(self, ax):
        color = [self.colors[self.outcomes[i]] if self.outcomes[i]>=0 else 'white'
                 for i in range(len(self.outcomes))]
        trials = np.array([k for k in self.neural_trials.keys()]).astype('int32')
        delay = np.array([self.neural_trials[t]['trial_delay']
                          for t in self.neural_trials.keys()])
        ax.scatter(trials, delay, color=color, alpha=1, s=5)
        ax.axhline(self.cate_delay, color='grey', lw=2, linestyle='--')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('trial id')
        ax.set_ylabel('delay (ms)')
        for i in range(len(self.states)):
            ax.plot([], color=self.colors[i], label=self.states[i])
        ax.legend(loc='center right')
        ax.set_title('2nd push delay setting')
    
    # trajectory aligned at 1st stimuli (short).
    def short_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_vis1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='vis1', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis1)
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('1st vis stim aligned trajectories (short)')
    
    # trajectory aligned at 1st pushing window end (short).
    def short_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_push1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='1st push onset', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 1st push onset (ms)')
        ax.set_title('1st push onset aligned trajectories (short)')

    # trajectory aligned at 1st retract (short).
    def short_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_retract1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='retract1', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 1st retract end (ms)')
        ax.set_title('1st retract end aligned trajectories (short)')
    
    # trajectory aligned at 2nd stimuli (short).
    def short_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_vis2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='vis2', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since 2nd vis stim (ms)')
        ax.set_title('2nd vis stim aligned trajectories (short)')
    
    # trajectory aligned at 2nd wait for push (short).
    def short_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_wait2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='wait2', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since 2nd push window onset (ms)')
        ax.set_title('2nd push window aligned trajectories (short)')
    
    # trajectory aligned at 2nd pushing window end (short).
    def short_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_push2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='2nd push onset', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 2nd push onset (ms)')
        ax.set_title('2nd push onset aligned trajectories (short)')
    
    # trajectory aligned at 2nd retract (short).
    def short_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_retract2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='retract1', linestyle='--')
        if len(align_data)>1:
            m, s = get_mean_sem(align_data[idx])
            upper = np.nanmax(m) + np.nanmax(s)
            lower = np.nanmin(m) - np.nanmax(s)
            self.plot_mean_sem(ax, align_time, m, s, 'mediumseagreen', 'reward')
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_motor), np.nanmax(self.neu_time_motor)])
            ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since 2nd retract onset (ms)')
        ax.set_title('2nd retract onset aligned trajectories (short)')
    
    # trajectory aligned at reward and punish (short).
    def short_align_pos_outcome(self, ax):
        [data_reward, time_reward, trial_delay_reward, _] = get_js_pos(self.neural_trials, 'trial_reward')
        [data_punish, time_punish, trial_delay_punish, _] = get_js_pos(self.neural_trials, 'trial_punish')
        idx_reward = get_trial_type(self.cate_delay, trial_delay_reward, 0)
        idx_punish = get_trial_type(self.cate_delay, trial_delay_punish, 0)
        mean_reward, sem_reward = get_mean_sem(data_reward[idx_reward])
        mean_punish, sem_punish = get_mean_sem(data_punish[idx_punish])
        l_idx = np.argmin(np.abs(time_reward - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_reward - self.neu_time_out[-1]))
        time_reward = time_reward[l_idx:r_idx]
        mean_reward = mean_reward[l_idx:r_idx]
        sem_reward  = sem_reward[l_idx:r_idx]
        l_idx = np.argmin(np.abs(time_punish - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_punish - self.neu_time_out[-1]))
        time_punish = time_punish[l_idx:r_idx]
        mean_punish = mean_punish[l_idx:r_idx]
        sem_punish  = sem_punish[l_idx:r_idx]
        sem_reward  = np.zeros_like(sem_reward) if np.isnan(np.sum(sem_reward)) else sem_reward
        sem_punish  = np.zeros_like(sem_punish) if np.isnan(np.sum(sem_punish)) else sem_punish
        upper = np.nanmax([mean_reward, mean_punish]) + np.nanmax([sem_reward, sem_punish])
        lower = -0.01
        ax.axvline(0, color='silver', lw=2, label='outcome', linestyle='--')
        self.plot_mean_sem(ax, time_reward, mean_reward, sem_reward, 'mediumseagreen', 'reward')
        self.plot_mean_sem(ax, time_punish, mean_punish, sem_punish, 'coral', 'punish')
        adjust_layout_js(ax)
        ax.set_xlabel('time since outcome (ms)')
        ax.set_xlim([np.nanmin(self.neu_time_out), np.nanmax(self.neu_time_out)])
        ax.set_ylim([lower, 1.1*upper])
        ax.set_title('outcome aligned trajectories (short)')
    
    # trajectory aligned at 1st stimuli (long).
    def long_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_vis1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='vis1', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis1)
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('1st vis stim aligned trajectories (long)')
    
    # trajectory aligned at 1st pushing window end (long).
    def long_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_push1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='1st push onset', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 1st push onset (ms)')
        ax.set_title('1st push onset aligned trajectories (long)')

    # trajectory aligned at 1st retract (long).
    def long_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_retract1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='retract1', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 1st retract end (ms)')
        ax.set_title('1st retract end aligned trajectories (long)')
    
    # trajectory aligned at 2nd stimuli (long).
    def long_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_vis2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='vis2', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since 2nd vis stim (ms)')
        ax.set_title('2nd vis stim aligned trajectories (long)')
    
    # trajectory aligned at 2nd wait for push (long).
    def long_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_wait2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='wait2', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since 2nd push window onset (ms)')
        ax.set_title('2nd push window aligned trajectories (long)')
    
    # trajectory aligned at 2nd pushing window end (long).
    def long_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_push2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='2nd push onset', linestyle='--')
        if len(align_data)>1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since 2nd push onset (ms)')
        ax.set_title('2nd push onset aligned trajectories (long)')
    
    # trajectory aligned at 2nd retract (long).
    def long_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(self.neural_trials, 'trial_retract2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='retract1', linestyle='--')
        if len(align_data)>1:
            m, s = get_mean_sem(align_data[idx])
            upper = np.nanmax(m) + np.nanmax(s)
            lower = np.nanmin(m) - np.nanmax(s)
            self.plot_mean_sem(ax, align_time, m, s, 'mediumseagreen', 'reward')
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_motor), np.nanmax(self.neu_time_motor)])
            ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since 2nd retract onset (ms)')
        ax.set_title('2nd retract onset aligned trajectories (long)')
    
    # trajectory aligned at reward and punish (long).
    def long_align_pos_outcome(self, ax):
        [data_reward, time_reward, trial_delay_reward, _] = get_js_pos(self.neural_trials, 'trial_reward')
        [data_punish, time_punish, trial_delay_punish, _] = get_js_pos(self.neural_trials, 'trial_punish')
        idx_reward = get_trial_type(self.cate_delay, trial_delay_reward, 1)
        idx_punish = get_trial_type(self.cate_delay, trial_delay_punish, 1)
        mean_reward, sem_reward = get_mean_sem(data_reward[idx_reward])
        mean_punish, sem_punish = get_mean_sem(data_punish[idx_punish])
        l_idx = np.argmin(np.abs(time_reward - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_reward - self.neu_time_out[-1]))
        time_reward = time_reward[l_idx:r_idx]
        mean_reward = mean_reward[l_idx:r_idx]
        sem_reward  = sem_reward[l_idx:r_idx]
        l_idx = np.argmin(np.abs(time_punish - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_punish - self.neu_time_out[-1]))
        time_punish = time_punish[l_idx:r_idx]
        mean_punish = mean_punish[l_idx:r_idx]
        sem_punish  = sem_punish[l_idx:r_idx]
        sem_reward  = np.zeros_like(sem_reward) if np.isnan(np.sum(sem_reward)) else sem_reward
        sem_punish  = np.zeros_like(sem_punish) if np.isnan(np.sum(sem_punish)) else sem_punish
        upper = np.nanmax([mean_reward, mean_punish]) + np.nanmax([sem_reward, sem_punish])
        lower = -0.01
        ax.axvline(0, color='silver', lw=2, label='outcome', linestyle='--')
        self.plot_mean_sem(ax, time_reward, mean_reward, sem_reward, 'mediumseagreen', 'reward')
        self.plot_mean_sem(ax, time_punish, mean_punish, sem_punish, 'coral', 'punish')
        adjust_layout_js(ax)
        ax.set_xlabel('time since outcome (ms)')
        ax.set_xlim([np.nanmin(self.neu_time_out), np.nanmax(self.neu_time_out)])
        ax.set_ylim([lower, 1.1*upper])
        ax.set_title('outcome aligned trajectories (long)')