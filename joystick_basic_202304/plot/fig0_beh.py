#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem

from modules.Alignment import get_js_pos
from plot.utils import adjust_layout_js


class plotter_all_beh:
    
    def __init__(
            self,
            neural_trials,
            ):
        self.neural_trials = neural_trials

    def plot_mean_sem(self, ax, t, m, s, c, l):
        ax.plot(t, m, color=c, label=l)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
    
    def plot_align_pos_outcome(self, ax, align_data, align_time, outcome):
        mean_reward = np.mean(align_data[outcome==1], axis=0)
        mean_punish = np.mean(align_data[outcome==-1], axis=0)
        sem_reward  = sem(align_data[outcome==1], axis=0)
        sem_punish  = sem(align_data[outcome==-1], axis=0)
        sem_reward = np.zeros_like(sem_reward) if np.isnan(np.sum(sem_reward)) else sem_reward
        sem_punish = np.zeros_like(sem_punish) if np.isnan(np.sum(sem_punish)) else sem_punish
        upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
        lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
        self.plot_mean_sem(ax, align_time, mean_reward, sem_reward, 'mediumseagreen', 'reward')
        self.plot_mean_sem(ax, align_time, mean_punish, sem_punish, 'coral', 'punish')
        adjust_layout_js(ax)
        ax.set_xlim([np.min(align_time), np.max(align_time)])
        ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        
    # outcome percentage.
    def outcome(self, ax):
        reward = np.array([not np.isnan(self.neural_trials[str(i)]['trial_reward'][0])
                  for i in range(len(self.neural_trials))])
        punish = np.array([not np.isnan(self.neural_trials[str(i)]['trial_punish'][0])
                  for i in range(len(self.neural_trials))])
        trial_types = np.array([self.neural_trials[str(i)]['trial_types']
                  for i in range(len(self.neural_trials))])
        reward_short = np.sum(reward[trial_types==1])
        punish_short = np.sum(punish[trial_types==1])
        reward_long  = np.sum(reward[trial_types==2])
        punish_long  = np.sum(punish[trial_types==2])
        ax.bar(
            0, reward_short/(reward_short+punish_short),
            bottom=0,
            edgecolor='white', width=0.25,
            color='mediumseagreen', label='reward')
        ax.bar(
            0, punish_short/(reward_short+punish_short),
            bottom=reward_short/(reward_short+punish_short),
            edgecolor='white', width=0.25,
            color='coral', label='punish')
        ax.bar(
            1, reward_long/(reward_long+punish_long),
            bottom=0,
            edgecolor='white', width=0.25,
            color='mediumseagreen', label='reward')
        ax.bar(
            1, punish_long/(reward_long+punish_long),
            bottom=reward_long/(reward_long+punish_long),
            edgecolor='white', width=0.25,
            color='coral', label='punish')
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(True)
        ax.set_xlabel('trial type')
        ax.set_ylabel('percentage')
        ax.set_xlim([-1,2])
        ax.set_xticks([0,1])
        ax.set_xticklabels(['short', 'long'])
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[0], handles[-1]]
        labels = [labels[0], labels[-1]]
        ax.legend(handles, labels, loc='upper right')
        ax.set_title('percentage of outcome for {} trials'.format(
            reward.shape[0]))
    
    # trajectory aligned at 1st stimuli.
    def align_pos_vis1(self, ax):
        [align_data, align_time, _, outcome] = get_js_pos(self.neural_trials, 'trial_vis1')
        if len(align_data) > 0:
            self.plot_align_pos_outcome(ax, align_data, align_time, outcome)
        ax.axvline(0, color='silver', lw=2, label='vis1', linestyle='--')
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('1st vis stim aligned trajectories')
    
    # trajectory aligned at 1st pressing window end.
    def align_pos_press1(self, ax):
        [align_data, align_time, _, outcome] = get_js_pos(self.neural_trials, 'trial_press1')
        if len(align_data) > 0:
            self.plot_align_pos_outcome(ax, align_data, align_time, outcome)
        ax.axvline(0, color='silver', lw=2, label='1st window end', linestyle='--')
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('1st pressing window aligned trajectories')

    # trajectory aligned at 1st retract.
    def align_pos_retract1(self, ax):
        [align_data, align_time, _, outcome] = get_js_pos(self.neural_trials, 'trial_retract1')
        if len(align_data) > 0:
            self.plot_align_pos_outcome(ax, align_data, align_time, outcome)
        ax.axvline(0, color='silver', lw=2, label='retract1', linestyle='--')
        ax.set_xlabel('time since 1st retract (ms)')
        ax.set_title('1st retract aligned trajectories')
    
    # trajectory aligned at 2nd stimuli.
    def align_pos_vis2(self, ax):
        [align_data, align_time, _, outcome] = get_js_pos(self.neural_trials, 'trial_vis2')
        if len(align_data) > 0:
            self.plot_align_pos_outcome(ax, align_data, align_time, outcome)
        adjust_layout_js(ax)
        ax.axvline(0, color='silver', lw=2, label='vis2', linestyle='--')
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('2nd vis stim aligned trajectories')
    
    # trajectory aligned at 2nd pressing window end.
    def align_pos_press2(self, ax):
        [align_data, align_time, _, outcome] = get_js_pos(self.neural_trials, 'trial_press2')
        if len(align_data) > 0:
            self.plot_align_pos_outcome(ax, align_data, align_time, outcome)
        ax.axvline(0, color='silver', lw=2, label='2nd window end', linestyle='--')
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_title('2nd pressing window aligned trajectories')
    
    # trajectory aligned at reward and punish.
    def align_pos_outcome(self, ax):
        [reward_data, reward_time, _, _] = get_js_pos(self.neural_trials, 'trial_reward')
        mean_reward = np.mean(reward_data, axis=0)
        sem_reward  = sem(reward_data, axis=0)
        [punish_data, punish_time, _, _] = get_js_pos(self.neural_trials, 'trial_punish')
        mean_punish = np.mean(punish_data, axis=0)
        sem_punish  = sem(punish_data, axis=0)
        upper = np.max(mean_reward) + np.max(sem_reward)
        lower = -0.1
        self.plot_mean_sem(ax, reward_time, mean_reward, sem_reward, 'mediumseagreen', 'reward')
        self.plot_mean_sem(ax, punish_time, mean_punish, sem_punish, 'coral', 'punish')
        ax.axvline(0, color='silver', lw=2, label='outcome', linestyle='--')
        adjust_layout_js(ax)
        ax.set_xlabel('time since reward (ms)')
        ax.set_xlim([np.min(reward_time), np.max(reward_time)])
        ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_title('outcome aligned trajectories')