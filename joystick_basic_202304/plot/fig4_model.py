#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from modules.Alignment import get_motor_response
from modules.Alignment import get_iti_response
from plot.utils import get_block_epoch
from plot.utils import get_trial_type
from plot.utils import get_roi_label_color
from plot.utils import utils
from plot.utils import adjust_layout_decode_box


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(labels)
        
        self.l_frames = 0
        self.r_frames = 5
        [self.neu_seq_vis1, _, self.outcome_vis1, _, self.delay_vis1] = get_stim_response(
                neural_trials, 'trial_vis1', self.l_frames, self.r_frames)
        [self.neu_seq_vis2, _, self.outcome_vis2, _, self.delay_vis2] = get_stim_response(
                neural_trials, 'trial_vis2', self.l_frames, self.r_frames)
        [self.neu_seq_push1, _, self.outcome_push1, self.delay_push1] = get_motor_response(
            neural_trials, 'trial_push1', self.l_frames, self.r_frames)
        [self.neu_seq_retract1, _, self.outcome_retract1, self.delay_retract1] = get_motor_response(
            neural_trials, 'trial_retract1', self.l_frames, self.r_frames)
        [self.neu_seq_wait2, _, self.outcome_wait2, self.delay_wait2] = get_motor_response(
            neural_trials, 'trial_wait2', self.l_frames, self.r_frames)
        [self.neu_seq_push2, _, self.outcome_push2, self.delay_push2] = get_motor_response(
            neural_trials, 'trial_push2', self.l_frames, self.r_frames)
        [self.neu_seq_reward, _, _, self.outcome_reward, self.delay_reward] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames, self.r_frames)
        [self.neu_seq_retract2, _, self.outcome_retract2, self.delay_retract2] = get_motor_response(
            neural_trials, 'trial_retract2', self.l_frames, self.r_frames)
        [self.neu_seq_iti, _, self.outcome_iti, self.delay_iti] = get_iti_response(
                neural_trials, self.l_frames, self.r_frames)
        self.significance = significance
        self.cate_delay = cate_delay
        
        self.offset = [-0.15, -0.05, 0.05, 0.15]
        self.c_all = 'mediumseagreen'
        self.c_chance = 'grey'
        self.state_all = [
            'vis1',
            'push1',
            'retract1',
            'vis2',
            'wait2',
            'push2',
            'reward',
            'retract2',
            'iti']
        self.significance = [
            significance['r_vis'],
            significance['r_push'],
            significance['r_retract'],
            significance['r_vis'],
            significance['r_wait'],
            significance['r_push'],
            significance['r_reward'],
            significance['r_retract'],
            np.ones_like(significance['r_vis']),
            ]
        self.neu_seq_all = [
            self.neu_seq_vis1,
            self.neu_seq_push1,
            self.neu_seq_retract1,
            self.neu_seq_vis2,
            self.neu_seq_wait2,
            self.neu_seq_push2,
            self.neu_seq_reward,
            self.neu_seq_retract2,
            self.neu_seq_iti]
        self.delay_all = [
            get_trial_type(self.cate_delay, self.delay_vis1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_push1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_retract1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_vis2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_wait2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_push2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_reward, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_retract2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_iti, 1).astype('int8')]
        self.outcome_all = [
            self.outcome_vis1,
            self.outcome_push1,
            self.outcome_retract1,
            self.outcome_vis2,
            self.outcome_wait2,
            self.outcome_push2,
            self.outcome_reward,
            self.outcome_retract2,
            self.outcome_iti]
    
    def plot_pop_decode_box(self, ax, x, y, outcome, pos, color, chance=True, reward_only=False):
        if reward_only:
            x = x[outcome==0,:,:]
            y = y[outcome==0]
        x = np.mean(x, axis=2)
        if chance:
            model = DummyClassifier(strategy="uniform")
        else:
            model = LogisticRegression(max_iter=20)#SVC(kernel='linear')
        loo = LeaveOneOut()
        acc = []
        for train_index, val_index in loo.split(x):
            try:
                x_train, x_val = x[train_index], x[val_index]
                y_train, y_val = y[train_index], y[val_index]
                model.fit(x_train, y_train)
                y_pred = model.predict(x_val)
                acc.append(accuracy_score(y_val, y_pred))
            except:
                pass
        acc = np.array(acc)
        m = np.mean(acc)
        s = sem(acc)
        ax.errorbar(
            pos,
            m, s,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=1)


class plotter_VIPTD_G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
        
    def plot_block_type_population_decoding(self, ax, ep):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                # find valid trial indice and block epoch indice.
                trial_idx, block_tran = get_block_epoch(self.delay_all[i])
                if ep == 'all':
                    idx = trial_idx
                    reward_only = True
                if ep == 'early':
                    idx = trial_idx * block_tran==0
                    reward_only = False
                if ep == 'late':
                    idx = trial_idx * block_tran==1
                    reward_only = False
                neu_seq = self.neu_seq_all[i][idx,:,:]
                delay = self.delay_all[i][idx]
                outcome = self.outcome_all[i][idx]
                # take valid trials.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, delay, outcome, i + self.offset[0], c_exc,
                    reward_only=reward_only)
                x = neu_seq[:,(self.labels==1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, delay, outcome, i + self.offset[1], c_inh,
                    reward_only=reward_only)
                x = neu_seq[:,self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, delay, outcome, i + self.offset[2], self.c_all,
                    reward_only=reward_only)
                x = neu_seq[:,self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, delay, outcome, i + self.offset[3], self.c_chance,
                    chance=True, reward_only=reward_only)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_all, label='all')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
    
    def plot_block_epoch_decoding_population(self, ax, block_type):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                # find valid trial indice and block epoch indice.
                trial_idx, block_tran = get_block_epoch(self.delay_all[i])
                if block_type == 'all':
                    idx = trial_idx
                if block_type == 'short':
                    idx = trial_idx * self.delay_all[i]==0
                if block_type == 'long':
                    idx = trial_idx * self.delay_all[i]==1
                # take valid trials.
                neu_seq = self.neu_seq_all[i][idx,:,:]
                block_tran = block_tran[idx]
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, self.outcome_all[i], i + self.offset[0], c_exc)
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, self.outcome_all[i], i + self.offset[1], c_inh)
                # all.
                x = neu_seq[:,self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, self.outcome_all[i], i + self.offset[2], self.c_all)
                # chance level.
                x = neu_seq[:,self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, self.outcome_all[i], i + self.offset[3], self.c_chance, chance=True)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_all, label='all')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
    
    def block_type_population_decode_all(self, ax):
        self.plot_block_type_population_decoding(ax, 'all')
        ax.set_title('population block decoding accuracy (reward) (all)')
    
    def block_type_population_decode_early(self, ax):
        self.plot_block_type_population_decoding(ax, 'early')
        ax.set_title('population block decoding accuracy (early)')
    
    def block_type_population_decode_late(self, ax):
        self.plot_block_type_population_decoding(ax, 'late')
        ax.set_title('population block decoding accuracy (late)')
    
    def block_tran_decode_all(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def block_tran_decode_short(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def block_tran_decode_long(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')


class plotter_L7G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)

    def plot_block_type_population_decoding(self, ax):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                x = self.neu_seq_all[i][:,(self.labels==-1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, self.delay_all[i], self.outcome_all[i], i + self.offset[1], c_exc)
                x = self.neu_seq_all[i][:,self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, self.delay_all[i], self.outcome_all[i], i + self.offset[2], self.c_chance, chance=True)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
        ax.set_title('population block decoding accuracy')
    
    def plot_block_epoch_decoding_population(self, ax, block_type):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                # find valid trial indice and block epoch indice.
                trial_idx, block_tran = get_block_epoch(self.delay_all[i])
                if block_type == 'all':
                    idx = trial_idx
                if block_type == 'short':
                    idx = trial_idx * self.delay_all[i]==0
                if block_type == 'long':
                    idx = trial_idx * self.delay_all[i]==1
                # take valid trials.
                neu_seq = self.neu_seq_all[i][idx,:,:]
                block_tran = block_tran[idx]
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, i + self.offset[0], c_exc)
                # chance level.
                x = neu_seq[:,self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, i + self.offset[3], self.c_chance, chance=True)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
    
    def block_tran_decode_all(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def block_tran_decode_short(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def block_tran_decode_long(self, ax):
        self.block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')


class plotter_VIPG8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
        
    def plot_block_type_population_decoding(self, ax):
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                x = self.neu_seq_all[i][:,(self.labels==1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, self.delay_all[i], self.outcome_all[i], i + self.offset[1], c_inh)
                x = self.neu_seq_all[i][:,self.significance[i],:]
                self.plot_pop_decode_box(
                    ax, x, self.delay_all[i], self.outcome_all[i], i + self.offset[3], self.c_chance, chance=True)
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
        ax.set_title('population block decoding accuracy')
    
    def plot_block_epoch_decoding_population(self, ax, block_type):
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                # find valid trial indice and block epoch indice.
                trial_idx, block_tran = get_block_epoch(self.delay_all[i])
                if block_type == 'all':
                    idx = trial_idx
                if block_type == 'short':
                    idx = trial_idx * self.delay_all[i]==0
                if block_type == 'long':
                    idx = trial_idx * self.delay_all[i]==1
                # take valid trials.
                neu_seq = self.neu_seq_all[i][idx,:,:]
                block_tran = block_tran[idx]
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, i + self.offset[1], c_inh)
                # chance level.
                x = neu_seq[:,self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    ax, x, block_tran, i + self.offset[3], self.c_chance, chance=True)
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_chance, label='shuffle')
        adjust_layout_decode_box(ax, self.state_all)
    
    def block_tran_decode_all(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def block_tran_decode_short(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def block_tran_decode_long(self, ax):
        self.plot_block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')