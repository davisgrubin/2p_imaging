#!/usr/bin/env python3

import numpy as np
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from modules.Alignment import run_get_perception_response

from utils import get_norm01_params
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_temporal_scaling_trial_multi_sess
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import apply_colormap
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 12))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance,
            temp_folder,
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(200*timescale)
        self.r_frames = int(200*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.list_significance = list_significance
        self.temp_folder = temp_folder
        self.alignment = {s: run_get_perception_response(
            self.temp_folder,
            self.list_neural_trials, s,
            self.l_frames, self.r_frames)
            for s in ['state_reward', 'state_punish', 'stim_seq']}

    def plot_stim(self, ax, side):
        xlim = [-500,3500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['stim_seq']['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['stim_seq']['neu_time'][l_idx:r_idx]
        trial_idx = [lt==side for lt in self.alignment['stim_seq']['list_trial_type']]
        [[color0, color1, color2, _],
         [neu_seq, _, stim_seq, _, _, _, _, outcome],
         [neu_labels, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment['stim_seq'], self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='perception')
        # plot results.
        neu_mean, neu_sem = get_mean_sem(neu_seq[:,l_idx:r_idx])
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for si in range(stim_seq.shape[0]):
            ax.fill_between(
                stim_seq[si,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color0, edgecolor='none', alpha=0.25, step='mid')
        # plot neural traces.
        self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color2, None)
        # adjust layouts.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim onset (ms)')
    
    def plot_stim_heatmap_neuron(self, ax, side):
        xlim = [-500,3500]
        win_sort = [-500, 500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['stim_seq']['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['stim_seq']['neu_time'][l_idx:r_idx]
        trial_idx = [lt==side for lt in self.alignment['stim_seq']['list_trial_type']]
        [[_, _, _, cmap],
         [neu_seq, _, stim_seq, _, _, _, _, outcome],
         [neu_labels, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment['stim_seq'], self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='perception')
        neu_seq = neu_seq[:,l_idx:r_idx]
        # plot heatmap.
        self.plot_heatmap_neuron(
                ax, neu_seq, neu_time, neu_seq, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq],
                cbar_label='dF/F')
        # adjust layout.
        ax.set_xlabel(f'time since stim onset (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')
    
    def plot_outcome(self, ax, out):
        xlim = [-1000,3000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['stim_seq']['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['state_'+out]['neu_time'][l_idx:r_idx]
        trial_idx = None
        [[color0, color1, color2, _],
         [neu_seq, _, _, _, _, _, _, _],
         [neu_labels, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment['state_'+out], self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='perception')
        # plot results.
        neu_mean, neu_sem = get_mean_sem(neu_seq[:,l_idx:r_idx])
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot outcome.
        ax.axvline(0, color=color0, lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color2, None)
        # adjust layouts.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel(f'time since {out} (ms)')

    def plot_outcome_heatmap_neuron(self, ax, out):
        xlim = [-500,3500]
        win_sort = [-500, 500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['stim_seq']['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['state_'+out]['neu_time'][l_idx:r_idx]
        trial_idx = None
        [[_, _, _, cmap],
         [neu_seq, _, _, _, _, _, _, _], _, _] = get_neu_trial(
            self.alignment['state_'+out], self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='perception')
        neu_seq = neu_seq[:,l_idx:r_idx]
        # plot heatmap.
        self.plot_heatmap_neuron(
                ax, neu_seq, neu_time, neu_seq, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq],
                cbar_label='dF/F')
        # adjust layout.
        ax.set_xlabel(f'time since {out} (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')
        
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names

    def perception(self, axs):
        try:
            
            self.plot_stim(axs[0][0], 0)
            axs[0][0].set_title('response to short interval')
            
            self.plot_stim(axs[0][1], 1)
            axs[0][1].set_title('response to long interval')
            
            self.plot_outcome(axs[0][2], 'reward')
            axs[0][2].set_title('response to reward')
            
            self.plot_outcome(axs[0][3], 'punish')
            axs[0][3].set_title('response to punish')
            
            self.plot_stim_heatmap_neuron(axs[1][0], 0)
            axs[1][0].set_title('response to short interval')
            
            self.plot_stim_heatmap_neuron(axs[1][1], 1)
            axs[1][1].set_title('response to long interval')
            
            self.plot_outcome_heatmap_neuron(axs[1][2], 'reward')
            axs[1][2].set_title('response to reward')
            
            self.plot_outcome_heatmap_neuron(axs[1][3], 'punish')
            axs[1][3].set_title('response to punish')

        except: pass
    