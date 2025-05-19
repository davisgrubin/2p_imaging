#!/usr/bin/env python3

import numpy as np

from utils import get_bin_stat
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 12))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials,
            temp_folder,
            ):
        super().__init__()
        self.n_sess = len(list_neural_trials)
        self.list_trial_labels = [nt['trial_labels'] for nt in list_neural_trials]
    
    def plot_psychometric(self, ax):
        isi_range = [0,2500]
        bin_size = 100
        # collect data.
        list_isi = [tl['isi'].to_numpy() for tl in self.list_trial_labels]
        list_decision_side = [tl['lick'].to_numpy() for tl in self.list_trial_labels]
        list_decision_side = [np.array([lds[li][1,0] for li in range(len(lds))]) for lds in list_decision_side]
        list_outcome = [tl['outcome'].to_numpy() for tl in self.list_trial_labels]
        # pool all sessions.
        list_isi = np.concatenate(list_isi)
        list_decision_side = np.concatenate(list_decision_side)
        list_outcome = np.concatenate(list_outcome)
        # exclude naive warmup.
        list_isi = list_isi[~np.isin(list_outcome, ['naive_reward', 'naive_punish'])]
        list_decision_side = list_decision_side[~np.isin(list_outcome, ['naive_reward', 'naive_punish'])]
        # compute stats.
        bin_x, bin_mean, bin_sem = get_bin_stat(list_isi, list_decision_side, isi_range, bin_size)
        # plot results.
        ax.plot(bin_x, bin_mean, color='#2C2C2C', linestyle='-', marker='.', markersize=4)
        ax.hlines(0.5, isi_range[0], isi_range[1], linestyle=':', color='grey')
        ax.vlines(np.nanmean(isi_range), 0.0, 1.0, linestyle=':', color='grey')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(isi_range)
        ax.set_ylim([-0.05,1.05])
        ax.set_xticks(np.arange(isi_range[1]/500+1)*500)
        ax.set_yticks(np.arange(5)*0.25)
        ax.set_xlabel('isi')
        ax.set_ylabel('right fraction')
        ax.set_title('psychometric function')

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, label_names, temp_folder):
        super().__init__(neural_trials, temp_folder)
        self.label_names = label_names
    
    def decision(self, axs):
        print('plotting decision results')
        try:
            self.plot_psychometric(axs[0])
        except: pass
        
