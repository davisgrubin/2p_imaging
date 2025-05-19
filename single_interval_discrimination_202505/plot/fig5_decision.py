#!/usr/bin/env python3

import numpy as np
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from modules.Alignment import run_get_lick_response

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
        self.alignment = run_get_lick_response(
            self.temp_folder,
            self.list_neural_trials,
            self.l_frames, self.r_frames)

    def plot_decision(self, ax):
        xlim = [-500,1500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # left.
        trial_idx = [(di==0)*(lt==1) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[color0, color1, color2, _],
         [neu_seq_left, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        # right.
        trial_idx = [(di==1)*(lt==1) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[color0, color1, color2, _],
         [neu_seq_right, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        # plot results.
        neu_meam_left,  neu_sem_left  = get_mean_sem(neu_seq_left[:,l_idx:r_idx])
        neu_mean_right, neu_sem_right = get_mean_sem(neu_seq_right[:,l_idx:r_idx])
        # find bounds.
        upper = np.nanmax([neu_meam_left, neu_mean_right]) + np.nanmax([neu_sem_left, neu_sem_right])
        lower = np.nanmin([neu_meam_left, neu_mean_right]) - np.nanmax([neu_sem_left, neu_sem_right])
        # plot licking.
        ax.axvline(0, color=color0, lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(ax, neu_time, neu_meam_left,  neu_sem_left, color1, None)
        self.plot_mean_sem(ax, neu_time, neu_mean_right, neu_sem_right, color2, None)
        # adjust layouts.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since decision licking (ms)')
        add_legend(ax, [color1,color2], ['left','right'], None, None, None, 'upper right')
    
    def plot_decision_heatmap_neuron(self, axs):
        xlim = [-500,1500]
        win_sort = [-500, 500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # left.
        trial_idx = [(di==0)*(lt==1) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[_, _, _, cmap],
         [neu_seq_left, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        neu_seq_left = neu_seq_left[:,l_idx:r_idx]
        # right.
        trial_idx = [(di==1)*(lt==1) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[_, _, _, cmap],
         [neu_seq_right, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        neu_seq_right = neu_seq_right[:,l_idx:r_idx]
        # plot heatmap.
        self.plot_heatmap_neuron(
                axs[0], neu_seq_left, neu_time, neu_seq_left, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_left],
                cbar_label='dF/F')
        self.plot_heatmap_neuron(
                axs[1], neu_seq_right, neu_time, neu_seq_right, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_right],
                cbar_label='dF/F')
        # adjust layout.
        axs[0].set_xlabel(f'time since decision licking (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')
        axs[1].set_xlabel(f'time since decision licking (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')

    def plot_consumption(self, ax):
        xlim = [-500,1500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # left.
        trial_idx = [(di==0)*(lt==0) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[color0, color1, color2, _],
         [neu_seq_left, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        # right.
        trial_idx = [(di==1)*(lt==0) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[color0, color1, color2, _],
         [neu_seq_right, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        # plot results.
        neu_meam_left,  neu_sem_left  = get_mean_sem(neu_seq_left[:,l_idx:r_idx])
        neu_mean_right, neu_sem_right = get_mean_sem(neu_seq_right[:,l_idx:r_idx])
        # find bounds.
        upper = np.nanmax([neu_meam_left, neu_mean_right]) + np.nanmax([neu_sem_left, neu_sem_right])
        lower = np.nanmin([neu_meam_left, neu_mean_right]) - np.nanmax([neu_sem_left, neu_sem_right])
        # plot licking.
        ax.axvline(0, color=color0, lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(ax, neu_time, neu_meam_left,  neu_sem_left, color1, None)
        self.plot_mean_sem(ax, neu_time, neu_mean_right, neu_sem_right, color2, None)
        # adjust layouts.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since consumption licking (ms)')
        add_legend(ax, [color1,color2], ['left','right'], None, None, None, 'upper right')
    
    def plot_consumption_heatmap_neuron(self, axs):
        xlim = [-500,1500]
        win_sort = [-500, 500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cate = [-1]
        # collect data.
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # left.
        trial_idx = [(di==0)*(lt==0) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[_, _, _, cmap],
         [neu_seq_left, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        neu_seq_left = neu_seq_left[:,l_idx:r_idx]
        # right.
        trial_idx = [(di==1)*(lt==0) for di,lt in zip(self.alignment['list_direction'], self.alignment['list_lick_type'])]
        [[_, _, _, cmap],
         [neu_seq_right, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            trial_idx=trial_idx,
            cate=cate, roi_id=None, mode='lick')
        neu_seq_right = neu_seq_right[:,l_idx:r_idx]
        # plot heatmap.
        self.plot_heatmap_neuron(
                axs[0], neu_seq_left, neu_time, neu_seq_left, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_left],
                cbar_label='dF/F')
        self.plot_heatmap_neuron(
                axs[1], neu_seq_right, neu_time, neu_seq_right, cmap, win_sort,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_right],
                cbar_label='dF/F')
        # adjust layout.
        axs[0].set_xlabel(f'time since consumption licking (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')
        axs[1].set_xlabel(f'time since consumption licking (ms) \n sorting window [{win_sort[0]},{win_sort[1]}] ms')


# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names

    def decision(self, axs):
        try:
            
            self.plot_decision(axs[0][0])
            axs[0][0].set_title('response to decision licking')
            
            self.plot_consumption(axs[0][1])
            axs[0][1].set_title('response to consumption licking')
            
            self.plot_decision_heatmap_neuron([axs[1][0],axs[2][0]])
            axs[1][0].set_title('response to decision licking (left)')
            axs[2][0].set_title('response to decision licking (right)')
            
            self.plot_consumption_heatmap_neuron([axs[1][1],axs[2][1]])
            axs[1][1].set_title('response to consumption licking (left)')
            axs[2][1].set_title('response to consumption licking (right)')

        except: pass
    