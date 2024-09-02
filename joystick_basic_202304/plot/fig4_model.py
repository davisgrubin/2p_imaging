#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from modules.Alignment import get_motor_response
from modules.Alignment import get_iti_response
from plot.utils import get_frame_idx_from_time
from plot.utils import get_block_epoch
from plot.utils import get_trial_type
from plot.utils import get_roi_label_color
from plot.utils import utils
from plot.utils import adjust_layout_decode_box
from plot.utils import adjust_layout_decode_outcome_pc


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(labels)

        self.win = [0, 160]
        [self.neu_seq_vis1, neu_time, self.outcome_vis1, _, self.delay_vis1] = get_stim_response(
                neural_trials, 'trial_vis1', 0, 50)
        [self.neu_seq_vis2, _, self.outcome_vis2, _, self.delay_vis2] = get_stim_response(
                neural_trials, 'trial_vis2', 0, 50)
        [self.neu_seq_push1, _, self.outcome_push1, self.delay_push1] = get_motor_response(
            neural_trials, 'trial_push1', 0, 50)
        [self.neu_seq_retract1, _, self.outcome_retract1, self.delay_retract1] = get_motor_response(
            neural_trials, 'trial_retract1', 0, 50)
        [self.neu_seq_wait2, _, self.outcome_wait2, self.delay_wait2] = get_motor_response(
            neural_trials, 'trial_wait2', 0, 50)
        [self.neu_seq_push2, _, self.outcome_push2, self.delay_push2] = get_motor_response(
            neural_trials, 'trial_push2', 0, 50)
        [self.neu_seq_reward, _, _, self.outcome_reward, self.delay_reward] = get_outcome_response(
                neural_trials, 'trial_reward', 0, 50)
        [self.neu_seq_retract2, _, self.outcome_retract2, self.delay_retract2] = get_motor_response(
            neural_trials, 'trial_retract2', 0, 50)
        [self.neu_seq_iti, _, self.outcome_iti, self.delay_iti] = get_iti_response(
                neural_trials, 0, 50)
        self.significance = significance
        self.cate_delay = cate_delay
        
        l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, self.win[0], self.win[1])
        self.offset = [-0.1, 0.0, 0.1]
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
            self.neu_seq_vis1[:,:,l_idx:r_idx],
            self.neu_seq_push1[:,:,l_idx:r_idx],
            self.neu_seq_retract1[:,:,l_idx:r_idx],
            self.neu_seq_vis2[:,:,l_idx:r_idx],
            self.neu_seq_wait2[:,:,l_idx:r_idx],
            self.neu_seq_push2[:,:,l_idx:r_idx],
            self.neu_seq_reward[:,:,l_idx:r_idx],
            self.neu_seq_retract2[:,:,l_idx:r_idx],
            self.neu_seq_iti[:,:,l_idx:r_idx]]
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
    
    def plot_pop_decode_box(self, ax, x, y, outcome, pos, color, chance=False, reward_only=False):
        if reward_only:
            x = x[outcome==0,:,:]
            y = y[outcome==0]
        x = np.mean(x, axis=2)
        if chance:
            model = DummyClassifier(strategy="uniform")
        else:
            model = SVC(kernel='linear')
        try:
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
                alpha=1 if not chance else 0.5,
                capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=1)
        except:
            pass
    
    def plot_class_outcome_pc(self, ax, y, outcome, pos, reward_only=False):
        if reward_only:
            y = y[outcome==0]
            outcome = outcome[outcome==0]
        bottom0 = 0
        bottom1 = 0
        for i in range(4):
            pc0 = np.sum((y==0)*(outcome==i)) / np.sum(y==0)
            pc1 = np.sum((y==1)*(outcome==i)) / np.sum(y==1)
            ax.bar(
                pos+self.offset[0], pc0,
                bottom=bottom0, edgecolor='white', width=2*np.abs(self.offset[0]),
                color=self.colors[i])
            ax.bar(
                pos+self.offset[2], pc1,
                bottom=bottom1, edgecolor='white', width=2*np.abs(self.offset[2]),
                color=self.colors[i])
            bottom0 += pc0
            bottom1 += pc1
    
    def plot_state_pca(self, ax, x, y, outcome, class_labels, reward_only=False, cate=None):
        if reward_only:
            x = x[outcome==0,:,:]
            y = y[outcome==0]
            outcome = outcome[outcome==0]
        x = np.mean(x, axis=2)
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        model = PCA(n_components=2)
        z = model.fit_transform(x)
        ax.scatter(z[y==0,0], z[y==0,1], color=color1)
        ax.scatter(z[y==1,0], z[y==1,1], color=color2)
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.scatter([], [], color=color1, label=class_labels[0])
        ax.scatter([], [], color=color2, label=class_labels[1])
        ax.legend(loc='upper right')
    
    def plot_low_dynamics(self, ax, x, y, outcome, class_labels, reward_only=False, cate=None):
        if reward_only:
            x = x[outcome==0,:,:]
            y = y[outcome==0]
            outcome = outcome[outcome==0]
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        x = np.transpose(x, (0,2,1))
        dim = x.shape
        model = PCA(n_components=2)
        z = model.fit_transform(x.reshape(-1, dim[-1])).reshape(dim[0], dim[1], 2)
        ax.plot(np.mean(z[y==0,:,0],axis=0), np.mean(z[y==0,:,1],axis=0), color=color1)
        ax.plot(np.mean(z[y==1,:,0],axis=0), np.mean(z[y==1,:,1],axis=0), color=color2)
        ax.scatter(np.mean(z[y==0,:,0],axis=0)[-1], np.mean(z[y==0,:,1],axis=0)[-1], color=color1)
        ax.scatter(np.mean(z[y==1,:,0],axis=0)[-1], np.mean(z[y==1,:,1],axis=0)[-1], color=color2)
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.plot([], [], color=color1, label=class_labels[0])
        ax.plot([], [], color=color2, label=class_labels[1])
        ax.legend(loc='upper right')
    
    def all_decode(self, axs):
        self.plot_block_type_population_decoding(axs[0], 'all')
        axs[0][0].set_title('population block decoding accuracy (reward) (all)')
        axs[0][1].set_title('outcome percentage (short|long)')
        self.plot_block_type_population_decoding(axs[1], 'early')
        axs[1][0].set_title('population block decoding accuracy (early)')
        axs[1][1].set_title('outcome percentage (short|long)')
        self.plot_block_type_population_decoding(axs[2], 'late')
        axs[2][0].set_title('population block decoding accuracy (late)')
        axs[2][1].set_title('outcome percentage (short|long)')
        self.plot_block_epoch_decoding_population(axs[3], 'all')
        axs[3][0].set_title('population block epoch decoding accuracy (all)')
        axs[3][1].set_title('outcome percentage (ep1|ep2)')
        self.plot_block_epoch_decoding_population(axs[4], 'short')
        axs[4][0].set_title('population block epoch decoding accuracy (short)')
        axs[4][1].set_title('outcome percentage (ep1|ep2)')
        self.plot_block_epoch_decoding_population(axs[5], 'long')
        axs[5][0].set_title('population block epoch decoding accuracy (long)')
        axs[5][1].set_title('outcome percentage (ep1|ep2)')

class plotter_VIPTD_G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
        
    def plot_block_type_population_decoding(self, axs, ep):
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
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[0], c_exc,
                    reward_only=reward_only)
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[0], c_exc,
                    chance=True, reward_only=reward_only)
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[1], c_inh,
                    reward_only=reward_only)
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[1], c_inh,
                    chance=True, reward_only=reward_only)
                # all.
                x = neu_seq[:,self.significance[i],:]
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[2], self.c_all,
                    reward_only=reward_only)
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[2], self.c_all,
                    chance=True, reward_only=reward_only)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], delay, outcome, i, reward_only=reward_only)
        axs[0].plot([], color=c_exc, label='exc')
        axs[0].plot([], color=c_inh, label='inh')
        axs[0].plot([], color=self.c_all, label='all')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        axs[0].set_xlabel('state [{},{}] ms window'.format(self.win[0], self.win[1]))
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def plot_block_epoch_decoding_population(self, axs, block_type):
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
                outcome = self.outcome_all[i][idx]
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[0], c_exc)
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[0], c_exc,
                    chance=True)
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[1], c_inh)
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[1], c_inh,
                    chance=True)
                # all.
                x = neu_seq[:,self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[2], self.c_all)
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[2], self.c_all,
                    chance=True)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], block_tran, outcome, i)
        axs[0].plot([], color=c_exc, label='exc')
        axs[0].plot([], color=c_inh, label='inh')
        axs[0].plot([], color=self.c_all, label='all')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def block_type_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[0], x, y, outcome, ['short','long'], reward_only=True, cate=-1)
        axs[0].set_title('PCA of block decoding features at WaitForPush2 (reward) (exc)')
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[1], x, y, outcome, ['short','long'], reward_only=True, cate=1)
        axs[1].set_title('PCA of block decoding features at WaitForPush2 (reward) (inh)')
        # all.
        x = neu_seq[:,self.significance[state],:].copy()
        self.plot_state_pca(axs[2], x, y, outcome, ['short','long'], reward_only=True, cate=0)
        axs[2].set_title('PCA of block decoding features at WaitForPush2 (reward) (all)')
    
    def block_tran_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[0], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=-1)
        axs[0].set_title('PCA of block epoch decoding features at WaitForPush2 (exc)')
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[1], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=1)
        axs[1].set_title('PCA of block epoch decoding features at WaitForPush2 (inh)')
        # all.
        x = neu_seq[:,self.significance[state],:].copy()
        self.plot_state_pca(axs[2], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=0)
        axs[2].set_title('PCA of block epoch decoding features at WaitForPush2 (all)')
    
    def block_type_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[0], x, y, outcome, ['short','long'], reward_only=True, cate=-1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (exc)')
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[1], x, y, outcome, ['short','long'], reward_only=True, cate=1)
        axs[1].set_title('PCA dynamics since WaitForPush2 (inh)')
        # all.
        x = neu_seq[:,self.significance[state],:].copy()
        self.plot_low_dynamics(axs[2], x, y, outcome, ['short','long'], reward_only=True, cate=0)
        axs[2].set_title('PCA dynamics since WaitForPush2 (all)')
    
    def block_tran_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[0], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=-1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (exc)')
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[1], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=1)
        axs[1].set_title('PCA dynamics since WaitForPush2 (inh)')
        # all.
        x = neu_seq[:,self.significance[state],:].copy()
        self.plot_low_dynamics(axs[2], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=0)
        axs[2].set_title('PCA dynamics since WaitForPush2 (all)')
        
class plotter_L7G8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)

    def plot_block_type_population_decoding(self, axs, ep):
        _, _, c_exc, _ = get_roi_label_color([-1], 0)
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
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[0], c_exc,
                    reward_only=reward_only)
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[0], c_exc,
                    chance=True, reward_only=reward_only)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], delay, outcome, i, reward_only=reward_only)
        axs[0].plot([], color=c_exc, label='exc')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        axs[0].set_xlabel('state [{},{}] ms window'.format(self.win[0], self.win[1]))
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def plot_block_epoch_decoding_population(self, axs, block_type):
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
                outcome = self.outcome_all[i][idx]
                # excitory.
                x = neu_seq[:,(self.labels==-1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[0], c_exc)
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[0], c_exc,
                    chance=True)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], block_tran, outcome, i)
        axs[0].plot([], color=c_exc, label='exc')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def block_type_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[0], x, y, outcome, ['short','long'], reward_only=True, cate=-1)
        axs[0].set_title('PCA of block decoding features at WaitForPush2 (reward) (exc)')
    
    def block_tran_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[0], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=-1)
        axs[0].set_title('PCA of block epoch decoding features at WaitForPush2 (exc)')
    
    def block_type_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[0], x, y, outcome, ['short','long'], reward_only=True, cate=-1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (exc)')
    
    def block_tran_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # excitory.
        x = neu_seq[:,(self.labels==-1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[0], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=-1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (exc)')
       

class plotter_VIPG8_model(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    def plot_block_type_population_decoding(self, axs, ep):
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
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:]
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[1], c_inh,
                    reward_only=reward_only)
                self.plot_pop_decode_box(
                    axs[0], x, delay, outcome, i + self.offset[1], c_inh,
                    chance=True, reward_only=reward_only)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], delay, outcome, i, reward_only=reward_only)
        axs[0].plot([], color=c_inh, label='inh')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        axs[0].set_xlabel('state [{},{}] ms window'.format(self.win[0], self.win[1]))
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def plot_block_epoch_decoding_population(self, axs, block_type):
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
                outcome = self.outcome_all[i][idx]
                # inhibitory.
                x = neu_seq[:,(self.labels==1)*self.significance[i],:].copy()
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[1], c_inh)
                self.plot_pop_decode_box(
                    axs[0], x, block_tran, outcome, i + self.offset[1], c_inh,
                    chance=True)
                # outcome percentage.
                self.plot_class_outcome_pc(axs[1], block_tran, outcome, i)
        axs[0].plot([], color=c_inh, label='inh')
        axs[0].plot([], color=self.c_chance, alpha=0.5, label='shuffle')
        adjust_layout_decode_box(axs[0], self.state_all)
        for i in range(len(self.states)):
            axs[1].plot([], color=self.colors[i], label=self.states[i])
        adjust_layout_decode_outcome_pc(axs[1], self.state_all)
    
    def block_type_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[1], x, y, outcome, ['short','long'], reward_only=True, cate=1)
        axs[0].set_title('PCA of block decoding features at WaitForPush2 (reward) (inh)')

    def block_tran_population_pca(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_all[state][trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_state_pca(axs[1], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=1)
        axs[0].set_title('PCA of block epoch decoding features at WaitForPush2 (inh)')
    
    def block_type_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, _ = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = self.delay_all[state][trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[1], x, y, outcome, ['short','long'], reward_only=True, cate=1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (inh)')
    
    def block_tran_dynamics(self, axs):
        state = 4
        # find valid trial indice and block epoch indice.
        trial_idx, block_tran = get_block_epoch(self.delay_all[state])
        neu_seq = self.neu_seq_wait2[trial_idx,:,:]
        y = block_tran[trial_idx]
        outcome = self.outcome_all[state][trial_idx]
        # inhibitory.
        x = neu_seq[:,(self.labels==1)*self.significance[state],:].copy()
        self.plot_low_dynamics(axs[1], x, y, outcome, ['ep1','ep2'], reward_only=True, cate=1)
        axs[0].set_title('PCA dynamics since WaitForPush2 (inh)')