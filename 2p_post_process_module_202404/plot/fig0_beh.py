#!/usr/bin/env python3

import numpy as np
from modules.Alignment import get_js_pos
from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from modules.Alignment import get_motor_response
from plot.utils import get_block_epoch
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
        self.l_frames_motor = 30
        self.r_frames_motor = 50
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
            trial_idx = idx*(outcome == i)
            if len(trial_idx) >= self.min_num_trial:
                m, s = get_mean_sem(align_data[trial_idx, :])
                m = m[l_idx:r_idx]
                s = s[l_idx:r_idx]
                self.plot_mean_sem(ax, align_time, m, s,
                                   self.colors[i], self.states[i])
                mean.append(m)
                sem.append(s)
        upper = np.nanmax(mean) + np.nanmax(sem)
        lower = -0.1
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(neu_time), np.nanmax(neu_time)])
        ax.set_ylim([lower, upper])

    def plot_align_pos_epoch(
            self, ax,
            align_data, align_time, outcome,
            delay, block,
            neu_time):
        l_idx = np.argmin(np.abs(align_time - neu_time[0]))
        r_idx = np.argmin(np.abs(align_time - neu_time[-1]))
        align_time = align_time[l_idx:r_idx]
        idx = get_trial_type(self.cate_delay, delay, block)
        trial_idx, block_tran = get_block_epoch(idx)
        if np.sum(outcome == 0) != 0:
            i_ep1 = (block_tran == 1) * trial_idx * idx * (outcome == 0)
            i_ep2 = (block_tran == 0) * trial_idx * idx * (outcome == 0)
        else:
            i_ep1 = (block_tran == 1) * trial_idx * idx * (outcome > 0)
            i_ep2 = (block_tran == 0) * trial_idx * idx * (outcome > 0)
        m_ep1, s_ep1 = get_mean_sem(align_data[i_ep1, :])
        m_ep2, s_ep2 = get_mean_sem(align_data[i_ep2, :])
        m_ep1 = m_ep1[l_idx:r_idx]
        s_ep1 = s_ep1[l_idx:r_idx]
        m_ep2 = m_ep2[l_idx:r_idx]
        s_ep2 = s_ep2[l_idx:r_idx]
        self.plot_mean_sem(ax, align_time, m_ep1, s_ep1, 'grey', 'ep1')
        self.plot_mean_sem(ax, align_time, m_ep2, s_ep2, self.colors[0], 'ep2')
        upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
        upper = upper if not np.isnan(upper) else 0
        lower = -0.1
        ax.set_ylim([lower, upper])
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(neu_time), np.nanmax(neu_time)])

    # outcome percentage.
    def session_outcome(self, ax):
        trial_delay = np.array([self.neural_trials[t]['trial_delay']
                                for t in self.neural_trials.keys()])
        trial_types = get_trial_type(
            self.cate_delay, trial_delay, 1).astype('int32')
        for block in [0, 1]:
            bottom = 0
            for i in range(4):
                pc = np.sum((trial_types == block) *
                            (self.outcomes == i)) / np.sum(trial_types == block)
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
        ax.set_xlim([-1, 3])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['short', 'long'])
        for i in range(len(self.states)):
            ax.plot([], color=self.colors[i], label=self.states[i])
        ax.legend(loc='upper right')
        ax.set_title('percentage of outcome for {} trials'.format(
            len(self.outcomes)))

    def all_short_align(self, axs):
        self.short_align_pos_vis1(axs[0])
        self.short_align_pos_push1(axs[1])
        self.short_align_pos_retract1(axs[2])
        self.short_align_pos_vis2(axs[3])
        self.short_align_pos_wait2(axs[4])
        self.short_align_pos_push2(axs[5])
        self.short_align_pos_reward(axs[6])
        self.short_align_pos_punish(axs[7])
        self.short_align_pos_retract2(axs[8])

    def all_long_align(self, axs):
        self.long_align_pos_vis1(axs[0])
        self.long_align_pos_push1(axs[1])
        self.long_align_pos_retract1(axs[2])
        self.long_align_pos_vis2(axs[3])
        self.long_align_pos_wait2(axs[4])
        self.long_align_pos_push2(axs[5])
        self.long_align_pos_reward(axs[6])
        self.long_align_pos_punish(axs[7])
        self.long_align_pos_retract2(axs[8])

    def all_short_epoch(self, axs):
        self.short_epoch_align_pos_vis1(axs[0])
        self.short_epoch_align_pos_push1(axs[1])
        self.short_epoch_align_pos_retract1(axs[2])
        self.short_epoch_align_pos_vis2(axs[3])
        self.short_epoch_align_pos_wait2(axs[4])
        self.short_epoch_align_pos_push2(axs[5])
        self.short_epoch_align_pos_reward(axs[6])
        self.short_epoch_align_pos_retract2(axs[7])

    def all_long_epoch(self, axs):
        self.long_epoch_align_pos_vis1(axs[0])
        self.long_epoch_align_pos_push1(axs[1])
        self.long_epoch_align_pos_retract1(axs[2])
        self.long_epoch_align_pos_vis2(axs[3])
        self.long_epoch_align_pos_wait2(axs[4])
        self.long_epoch_align_pos_push2(axs[5])
        self.long_epoch_align_pos_reward(axs[6])
        self.long_epoch_align_pos_retract2(axs[7])

    # delay curve.
    def delay_dist(self, ax):
        color = [self.colors[self.outcomes[i]] if self.outcomes[i] >= 0 else 'white'
                 for i in range(len(self.outcomes))]
        trials = np.array(
            [k for k in self.neural_trials.keys()]).astype('int32')
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

    # all push onset.
    def onset(self, ax):
        [data_p1, time_p1, _, outcome_p1] = get_js_pos(
            self.neural_trials, 'trial_push1')
        [data_p2, time_p2, _, outcome_p2] = get_js_pos(
            self.neural_trials, 'trial_push2')
        ax.axvline(0, color='grey', lw=1, label='PushOnset', linestyle='--')
        l_idx_1 = np.argmin(np.abs(time_p1 - self.neu_time_motor[0]))
        r_idx_1 = np.argmin(np.abs(time_p1 - self.neu_time_motor[-1]))
        l_idx_2 = np.argmin(np.abs(time_p2 - self.neu_time_motor[0]))
        r_idx_2 = np.argmin(np.abs(time_p2 - self.neu_time_motor[-1]))
        time_p1 = time_p1[l_idx_1:r_idx_1]
        time_p2 = time_p2[l_idx_2:r_idx_2]
        m_push1_all,    s_push1_all = get_mean_sem(data_p1[outcome_p1 != 1, :])
        m_push2_reward, s_push2_reward = get_mean_sem(
            data_p2[outcome_p2 == 0, :])
        m_push2_punish, s_push2_punish = get_mean_sem(
            data_p2[outcome_p2 == 3, :])
        m_push1_all = m_push1_all[l_idx_1:r_idx_1]
        s_push1_all = s_push1_all[l_idx_1:r_idx_1]
        m_push2_reward = m_push2_reward[l_idx_2:r_idx_2]
        s_push2_reward = s_push2_reward[l_idx_2:r_idx_2]
        m_push2_punish = m_push2_punish[l_idx_2:r_idx_2]
        s_push2_punish = s_push2_punish[l_idx_2:r_idx_2]
        self.plot_mean_sem(ax, time_p1, m_push1_all,
                           s_push1_all,   'grey',          'push1 all')
        self.plot_mean_sem(ax, time_p2, m_push2_reward,
                           s_push2_reward, self.colors[0], 'push2 reward')
        self.plot_mean_sem(ax, time_p2, m_push2_punish,
                           s_push2_punish, self.colors[3], 'push2 early')
        upper = np.nanmax([m_push1_all, m_push2_reward, m_push2_punish]) +\
            np.nanmax([s_push1_all, s_push2_reward, s_push2_punish])
        upper = 0 if np.isnan(upper) else upper
        lower = -0.1
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(self.neu_time_motor),
                    np.nanmax(self.neu_time_motor)])
        ax.set_ylim([lower, upper])
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('push onset aligned trajectories')

    # trajectory aligned at Vis1 (short).
    def short_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='Vis1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('Vis1 aligned trajectories (short)')

    # trajectory aligned at PushOnset1 (short).
    def short_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='PushOnset1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('PushOnset1 aligned trajectories (short)')

    # trajectory aligned at Retract1 end (short).
    def short_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract1')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='Retract1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('Retract1 end aligned trajectories (short)')

    # trajectory aligned at Vis2 (short).
    def short_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='Vis2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('Vis2 aligned trajectories (short)')

    # trajectory aligned at WaitForPush2 start (short).
    def short_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_wait2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2,
                   label='WaitForPush2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('WaitForPush2 start aligned trajectories (short)')

    # trajectory aligned at PushOnset2 (short).
    def short_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='PushOnset2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('PushOnset2 aligned trajectories (short)')

    # trajectory aligned at Retract2 (short).
    def short_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract2')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='Retract2', linestyle='--')
        if len(align_data) > 1:
            m, s = get_mean_sem(align_data[idx])
            upper = np.nanmax(m) + np.nanmax(s)
            lower = np.nanmin(m) - np.nanmax(s)
            self.plot_mean_sem(ax, align_time, m, s, self.colors[0], 'reward')
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_motor),
                        np.nanmax(self.neu_time_motor)])
            ax.set_ylim([lower, upper])
        ax.set_xlabel('time since Retract2 start (ms)')
        ax.set_title('Retract2 start aligned trajectories (short)')

    # trajectory aligned at reward (short).
    def short_align_pos_reward(self, ax):
        [data_reward, time_reward, trial_delay_reward, _] = get_js_pos(
            self.neural_trials, 'trial_reward')
        idx_reward = get_trial_type(self.cate_delay, trial_delay_reward, 0)
        mean_reward, sem_reward = get_mean_sem(data_reward[idx_reward])
        l_idx = np.argmin(np.abs(time_reward - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_reward - self.neu_time_out[-1]))
        time_reward = time_reward[l_idx:r_idx]
        mean_reward = mean_reward[l_idx:r_idx]
        sem_reward = sem_reward[l_idx:r_idx]
        sem_reward = np.zeros_like(sem_reward) if np.isnan(
            np.sum(sem_reward)) else sem_reward
        upper = np.nanmax(mean_reward) + np.nanmax(sem_reward)
        lower = -0.01
        ax.axvline(0, color='silver', lw=2, label='reward', linestyle='--')
        self.plot_mean_sem(ax, time_reward, mean_reward,
                           sem_reward, self.colors[0], 'reward')
        adjust_layout_js(ax)
        ax.set_xlabel('time since reward (ms)')
        ax.set_xlim([np.nanmin(self.neu_time_out),
                    np.nanmax(self.neu_time_out)])
        ax.set_ylim([lower, upper])
        ax.set_title('reward aligned trajectories (short)')

    # trajectory aligned at punish (short).
    def short_align_pos_punish(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_punish')
        idx = get_trial_type(self.cate_delay, trial_delay, 0)
        ax.axvline(0, color='silver', lw=2, label='punish', linestyle='--')
        if len(align_data) > 1:
            l_idx = np.argmin(np.abs(align_time - self.neu_time_out[0]))
            r_idx = np.argmin(np.abs(align_time - self.neu_time_out[-1]))
            align_time = align_time[l_idx:r_idx]
            mean = []
            sem = []
            for i in [1, 2, 3]:
                trial_idx = idx*(outcome == i)
                if len(trial_idx) >= self.min_num_trial:
                    m, s = get_mean_sem(align_data[trial_idx, :])
                    m = m[l_idx:r_idx]
                    s = s[l_idx:r_idx]
                    self.plot_mean_sem(ax, align_time, m, s,
                                       self.colors[i], self.states[i])
                    mean.append(m)
                    sem.append(s)
            upper = np.nanmax(mean) + np.nanmax(sem)
            lower = -0.1
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_out),
                        np.nanmax(self.neu_time_out)])
            ax.set_ylim([lower, upper])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('punish aligned trajectories (short)')

    # trajectory aligned at Vis1 with epoch (short).
    def short_epoch_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis1')
        ax.axvline(0, color='silver', lw=2, label='Vis1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_vis1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('Vis1 aligned trajectories (short)')

    # trajectory aligned at PushOnset1 with epoch (short).
    def short_epoch_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push1')
        ax.axvline(0, color='silver', lw=2, label='PushOnset1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_motor)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('PushOnset1 aligned trajectories (short)')

    # trajectory aligned at Retract1 end with epoch (short).
    def short_epoch_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract1')
        ax.axvline(0, color='silver', lw=2, label='Retract1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_motor)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('Retract1 end aligned trajectories (short)')

    # trajectory aligned at Vis2 with epoch (short).
    def short_epoch_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis2')
        ax.axvline(0, color='silver', lw=2, label='Vis2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('Vis2 aligned trajectories (short)')

    # trajectory aligned at WaitForPush2 start with epoch (short).
    def short_epoch_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_wait2')
        ax.axvline(0, color='silver', lw=2,
                   label='WaitForPush2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('WaitForPush2 start aligned trajectories (short)')

    # trajectory aligned at PushOnset2 with epoch (short).
    def short_epoch_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push2')
        ax.axvline(0, color='silver', lw=2, label='PushOnset2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_motor)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('PushOnset2 aligned trajectories (short)')

    # trajectory aligned at Retract2 with epoch (short).
    def short_epoch_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract2')
        ax.axvline(0, color='silver', lw=2, label='Retract2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 0,
                self.neu_time_motor)
        ax.set_xlabel('time since Retract2 start (ms)')
        ax.set_title('Retract2 start aligned trajectories (short)')

    # trajectory aligned at reward with epoch (short).
    def short_epoch_align_pos_reward(self, ax):
        [align_data, align_time, trial_delay_reward, _] = get_js_pos(
            self.neural_trials, 'trial_reward')
        idx = get_trial_type(self.cate_delay, trial_delay_reward, 0)
        trial_idx, block_tran = get_block_epoch(idx)
        l_idx = np.argmin(np.abs(align_time - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(align_time - self.neu_time_out[-1]))
        align_time = align_time[l_idx:r_idx]
        i_ep1 = (block_tran == 1) * trial_idx * idx
        i_ep2 = (block_tran == 0) * trial_idx * idx
        m_ep1, s_ep1 = get_mean_sem(align_data[i_ep1, :])
        m_ep2, s_ep2 = get_mean_sem(align_data[i_ep2, :])
        m_ep1 = m_ep1[l_idx:r_idx]
        s_ep1 = s_ep1[l_idx:r_idx]
        m_ep2 = m_ep2[l_idx:r_idx]
        s_ep2 = s_ep2[l_idx:r_idx]
        ax.axvline(0, color='silver', lw=2, label='reward', linestyle='--')
        if not np.isnan(np.sum(m_ep1)) and not np.isnan(np.sum(m_ep2)):
            self.plot_mean_sem(ax, align_time, m_ep1, s_ep1, 'grey', 'ep1')
            self.plot_mean_sem(ax, align_time, m_ep2,
                               s_ep2, self.colors[0], 'ep2')
            upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
            lower = np.nanmin([m_ep1, m_ep2]) - np.nanmax([s_ep1, s_ep2])
            lower = -0.1
            ax.set_ylim([lower, upper])
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(self.neu_time_out),
                    np.nanmax(self.neu_time_out)])
        ax.set_title('reward aligned trajectories (short)')

    # trajectory aligned at Vis1 (long).
    def long_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='Vis1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('Vis1 aligned trajectories (long)')

    # trajectory aligned at PushOnset1 (long).
    def long_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='PushOnset1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('PushOnset1 aligned trajectories (long)')

    # trajectory aligned at Retract1 (long).
    def long_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract1')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='Retract1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since Retract1 (ms)')
        ax.set_title('Retract1 aligned trajectories (long)')

    # trajectory aligned at Vis2 (long).
    def long_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='Vis2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('Vis2 aligned trajectories (long)')

    # trajectory aligned at WaitForPush2 (long).
    def long_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_wait2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2,
                   label='WaitForPush2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since WaitForPush2 (ms)')
        ax.set_title('WaitForPush2 aligned trajectories (long)')

    # trajectory aligned at PushOnset2 (long).
    def long_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='PushOnset2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_outcome(
                ax, align_data, align_time, outcome, idx, self.neu_time_motor)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('PushOnset2 aligned trajectories (long)')

    # trajectory aligned at Retract2 (long).
    def long_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract2')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='Retract2', linestyle='--')
        if len(align_data) > 1:
            m, s = get_mean_sem(align_data[idx])
            upper = np.nanmax(m) + np.nanmax(s)
            lower = np.nanmin(m) - np.nanmax(s)
            self.plot_mean_sem(ax, align_time, m, s, self.colors[0], 'reward')
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_motor),
                        np.nanmax(self.neu_time_motor)])
            ax.set_ylim([lower, upper])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('Retract2 aligned trajectories (long)')

    # trajectory aligned at reward (long).
    def long_align_pos_reward(self, ax):
        [data_reward, time_reward, trial_delay_reward, _] = get_js_pos(
            self.neural_trials, 'trial_reward')
        idx_reward = get_trial_type(self.cate_delay, trial_delay_reward, 1)
        mean_reward, sem_reward = get_mean_sem(data_reward[idx_reward])
        l_idx = np.argmin(np.abs(time_reward - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(time_reward - self.neu_time_out[-1]))
        time_reward = time_reward[l_idx:r_idx]
        mean_reward = mean_reward[l_idx:r_idx]
        sem_reward = sem_reward[l_idx:r_idx]
        sem_reward = np.zeros_like(sem_reward) if np.isnan(
            np.sum(sem_reward)) else sem_reward
        upper = np.nanmax(mean_reward) + np.nanmax(sem_reward)
        lower = -0.01
        ax.axvline(0, color='silver', lw=2, label='reward', linestyle='--')
        self.plot_mean_sem(ax, time_reward, mean_reward,
                           sem_reward, self.colors[0], 'reward')
        adjust_layout_js(ax)
        ax.set_xlabel('time since reward (ms)')
        ax.set_xlim([np.nanmin(self.neu_time_out),
                    np.nanmax(self.neu_time_out)])
        ax.set_ylim([lower, upper])
        ax.set_title('reward aligned trajectories (short)')

    # trajectory aligned at punish (long).
    def long_align_pos_punish(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_punish')
        idx = get_trial_type(self.cate_delay, trial_delay, 1)
        ax.axvline(0, color='silver', lw=2, label='punish', linestyle='--')
        if len(align_data) > 1:
            l_idx = np.argmin(np.abs(align_time - self.neu_time_out[0]))
            r_idx = np.argmin(np.abs(align_time - self.neu_time_out[-1]))
            align_time = align_time[l_idx:r_idx]
            mean = []
            sem = []
            for i in [1, 2, 3]:
                trial_idx = idx*(outcome == i)
                if len(trial_idx) >= self.min_num_trial:
                    m, s = get_mean_sem(align_data[trial_idx, :])
                    m = m[l_idx:r_idx]
                    s = s[l_idx:r_idx]
                    self.plot_mean_sem(ax, align_time, m, s,
                                       self.colors[i], self.states[i])
                    mean.append(m)
                    sem.append(s)
            upper = np.nanmax(mean) + np.nanmax(sem)
            lower = -0.1
            adjust_layout_js(ax)
            ax.set_xlim([np.nanmin(self.neu_time_out),
                        np.nanmax(self.neu_time_out)])
            ax.set_ylim([lower, upper])
        ax.set_xlabel('time since punish (ms)')
        ax.set_title('punish aligned trajectories (long)')

    # trajectory aligned at Vis1 with epoch (long).
    def long_epoch_align_pos_vis1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis1')
        ax.axvline(0, color='silver', lw=2, label='Vis1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_vis1)
        ax.set_xlabel('time since Vis1 (ms)')
        ax.set_title('Vis1 aligned trajectories (long)')

    # trajectory aligned at PushOnset1 with epoch (long).
    def long_epoch_align_pos_push1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push1')
        ax.axvline(0, color='silver', lw=2, label='PushOnset1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_motor)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('PushOnset1 aligned trajectories (long)')

    # trajectory aligned at Retract1 end with epoch (long).
    def long_epoch_align_pos_retract1(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract1')
        ax.axvline(0, color='silver', lw=2, label='Retract1', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_motor)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('Retract1 end aligned trajectories (long)')

    # trajectory aligned at Vis2 with epoch (long).
    def long_epoch_align_pos_vis2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_vis2')
        ax.axvline(0, color='silver', lw=2, label='Vis2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_vis2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since Vis2 (ms)')
        ax.set_title('Vis2 aligned trajectories (long)')

    # trajectory aligned at WaitForPush2 start with epoch (long).
    def long_epoch_align_pos_wait2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_wait2')
        ax.axvline(0, color='silver', lw=2,
                   label='WaitForPush2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_wait2)
        adjust_layout_js(ax)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('WaitForPush2 start aligned trajectories (long)')

    # trajectory aligned at PushOnset2 with epoch (long).
    def long_epoch_align_pos_push2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_push2')
        ax.axvline(0, color='silver', lw=2, label='PushOnset2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_motor)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('PushOnset2 aligned trajectories (long)')

    # trajectory aligned at Retract2 with epoch (long).
    def long_epoch_align_pos_retract2(self, ax):
        [align_data, align_time, trial_delay, outcome] = get_js_pos(
            self.neural_trials, 'trial_retract2')
        ax.axvline(0, color='silver', lw=2, label='Retract2', linestyle='--')
        if len(align_data) > 1:
            self.plot_align_pos_epoch(
                ax, align_data, align_time, outcome,
                trial_delay, 1,
                self.neu_time_motor)
        ax.set_xlabel('time since Retract2 start (ms)')
        ax.set_title('Retract2 start aligned trajectories (long)')

    # trajectory aligned at reward with epoch (long).
    def long_epoch_align_pos_reward(self, ax):
        [align_data, align_time, trial_delay_reward, _] = get_js_pos(
            self.neural_trials, 'trial_reward')
        idx = get_trial_type(self.cate_delay, trial_delay_reward, 0)
        trial_idx, block_tran = get_block_epoch(idx)
        l_idx = np.argmin(np.abs(align_time - self.neu_time_out[0]))
        r_idx = np.argmin(np.abs(align_time - self.neu_time_out[-1]))
        align_time = align_time[l_idx:r_idx]
        i_ep1 = (block_tran == 1) * trial_idx * idx
        i_ep2 = (block_tran == 0) * trial_idx * idx
        m_ep1, s_ep1 = get_mean_sem(align_data[i_ep1, :])
        m_ep2, s_ep2 = get_mean_sem(align_data[i_ep2, :])
        m_ep1 = m_ep1[l_idx:r_idx]
        s_ep1 = s_ep1[l_idx:r_idx]
        m_ep2 = m_ep2[l_idx:r_idx]
        s_ep2 = s_ep2[l_idx:r_idx]
        ax.axvline(0, color='silver', lw=2, label='reward', linestyle='--')
        if not np.isnan(np.sum(m_ep1)) and not np.isnan(np.sum(m_ep2)):
            self.plot_mean_sem(ax, align_time, m_ep1, s_ep1, 'grey', 'ep1')
            self.plot_mean_sem(ax, align_time, m_ep2,
                               s_ep2, self.colors[0], 'ep2')
            upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
            lower = np.nanmin([m_ep1, m_ep2]) - np.nanmax([s_ep1, s_ep2])
            lower = -0.1
            ax.set_ylim([lower, upper])
        adjust_layout_js(ax)
        ax.set_xlim([np.nanmin(self.neu_time_out),
                    np.nanmax(self.neu_time_out)])
        ax.set_title('reward aligned trajectories (long)')
