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


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(labels)
        
        self.l_frames = 5
        self.r_frames = 15
        [self.neu_seq_vis1, _, _, _, self.delay_vis1] = get_stim_response(
                neural_trials, 'trial_vis1', self.l_frames, self.r_frames)
        [self.neu_seq_vis2, _, _, _, self.delay_vis2] = get_stim_response(
                neural_trials, 'trial_vis2', self.l_frames, self.r_frames)
        [self.neu_seq_push1, _, _, self.delay_push1] = get_motor_response(
            neural_trials, 'trial_push1', self.l_frames, self.r_frames)
        [self.neu_seq_retract1, _, _, self.delay_retract1] = get_motor_response(
            neural_trials, 'trial_retract1', self.l_frames, self.r_frames)
        [self.neu_seq_wait2, _, _, self.delay_wait2] = get_motor_response(
            neural_trials, 'trial_wait2', self.l_frames, self.r_frames)
        [self.neu_seq_push2, _, _, self.delay_push2] = get_motor_response(
            neural_trials, 'trial_push2', self.l_frames, self.r_frames)
        [self.neu_seq_retract2, _, _, self.delay_retract2] = get_motor_response(
            neural_trials, 'trial_retract2', self.l_frames, self.r_frames)
        [self.neu_seq_reward, _, _, _, self.delay_reward] = get_outcome_response(
                neural_trials, 'trial_reward', self.l_frames, self.r_frames)
        [self.neu_seq_punish, _, _, _, self.delay_punish] = get_outcome_response(
                neural_trials, 'trial_punish', self.l_frames, self.r_frames)
        [self.neu_seq_iti, _, self.delay_iti] = get_iti_response(
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
            'retract2',
            'reward',
            'punish',
            'iti']
        self.significance = [
            significance['r_vis'],
            significance['r_push'],
            significance['r_retract'],
            significance['r_vis'],
            significance['r_wait'],
            significance['r_push'],
            significance['r_retract'],
            significance['r_reward'],
            significance['r_punish'],
            np.ones_like(significance['r_vis']),
            ]
        self.neu_seq_all = [
            self.neu_seq_vis1,
            self.neu_seq_push1,
            self.neu_seq_retract1,
            self.neu_seq_vis2,
            self.neu_seq_wait2,
            self.neu_seq_push2,
            self.neu_seq_retract2,
            self.neu_seq_reward,
            self.neu_seq_punish,
            self.neu_seq_iti]
        self.delay_all = [
            get_trial_type(self.cate_delay, self.delay_vis1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_push1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_retract1, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_vis2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_wait2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_push2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_retract2, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_reward, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_punish, 1).astype('int8'),
            get_trial_type(self.cate_delay, self.delay_iti, 1).astype('int8')]
        
    def run_pop_decode(self, x, y, chance=False):
        if chance:
            model = DummyClassifier(strategy="uniform")
        else:
            model = LogisticRegression()#SVC(kernel='linear') 
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
        return m, s
        
class plotter_VIPTD_G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
        
    def plot_block_type_population_decoding(self, ax):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                neu_exc = self.neu_seq_all[i][:,(self.labels==-1)*self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_exc, axis=2), self.delay_all[i])
                ax.errorbar(
                    i + self.offset[0],
                    m, s,
                    color=c_exc,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                neu_inh = self.neu_seq_all[i][:,(self.labels==1)*self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_inh, axis=2), self.delay_all[i])
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_inh,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                neu_all = self.neu_seq_all[i][:,self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), self.delay_all[i])
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_all,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                neu_all = self.neu_seq_all[i][:,self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), self.delay_all[i], chance=True)
                ax.errorbar(
                    i + self.offset[3],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_all, label='all')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
        ax.set_title('population block decoding accuracy')
    
    def block_epoch_decoding_population(self, ax, block_type):
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
                neu_exc = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_exc, axis=2), block_tran)
                ax.errorbar(
                    i + self.offset[0],
                    m, s,
                    color=c_exc,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                # inhibitory.
                neu_inh = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_inh, axis=2), block_tran)
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_inh,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                # all.
                neu_all = neu_seq[:,self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), block_tran)
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_all,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                # chance level.
                neu_all = neu_seq[:,self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), block_tran, chance=True)
                ax.errorbar(
                    i + self.offset[3],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_all, label='all')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
    
    def plot_block_tran_decode_all(self, ax):
        self.block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def plot_block_tran_decode_short(self, ax):
        self.block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def plot_block_tran_decode_long(self, ax):
        self.block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')


class plotter_L7G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)

    def plot_block_type_population_decoding(self, ax):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                neu_exc = self.neu_seq_all[i][:,(self.labels==-1)*self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_exc, axis=2), self.delay_all[i])
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_exc,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                neu_all = self.neu_seq_all[i][:,self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), self.delay_all[i], chance=True)
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
        ax.set_title('population block decoding accuracy')
    
    def block_epoch_decoding_population(self, ax, block_type):
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
                neu_exc = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_exc, axis=2), block_tran)
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_exc,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                # chance level.
                neu_all = neu_seq[:,self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), block_tran, chance=True)
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_exc, label='exc')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
    
    def plot_block_tran_decode_all(self, ax):
        self.block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def plot_block_tran_decode_short(self, ax):
        self.block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def plot_block_tran_decode_long(self, ax):
        self.block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')


class plotter_VIPG8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
        
    def plot_block_type_population_decoding(self, ax):
        _, _, c_inh, _ = get_roi_label_color([1], 0)
        for i in range(len(self.state_all)):
            if not np.isnan(np.sum(self.neu_seq_all[i])):
                neu_inh = self.neu_seq_all[i][:,(self.labels==1)*self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_inh, axis=2), self.delay_all[i])
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_inh,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                neu_all = self.neu_seq_all[i][:,self.significance[i],:]
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), self.delay_all[i], chance=True)
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
        ax.set_title('population block decoding accuracy')
    
    def block_epoch_decoding_population(self, ax, block_type):
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
                neu_inh = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_inh, axis=2), block_tran)
                ax.errorbar(
                    i + self.offset[1],
                    m, s,
                    color=c_inh,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
                # chance level.
                neu_all = neu_seq[:,self.significance[i],:].copy()
                m, s = self.run_pop_decode(np.mean(neu_all, axis=2), block_tran, chance=True)
                ax.errorbar(
                    i + self.offset[2],
                    m, s,
                    color=self.c_chance,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=1)
        ax.plot([], color=c_inh, label='inh')
        ax.plot([], color=self.c_chance, label='shuffle')
        ax.legend(loc='upper right')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-0.5, len(self.state_all)+1])
        ax.set_xlabel('state')
        ax.set_ylabel('validation accuracy')
        ax.set_xticks(np.arange(len(self.state_all)))
        ax.set_xticklabels(self.state_all)
    
    def plot_block_tran_decode_all(self, ax):
        self.block_epoch_decoding_population(ax, 'all')
        ax.set_title('population block epoch decoding accuracy (all)')
    
    def plot_block_tran_decode_short(self, ax):
        self.block_epoch_decoding_population(ax, 'short')
        ax.set_title('population block epoch decoding accuracy (short)')
    
    def plot_block_tran_decode_long(self, ax):
        self.block_epoch_decoding_population(ax, 'long')
        ax.set_title('population block epoch decoding accuracy (long)')